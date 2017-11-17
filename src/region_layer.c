#include "region_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>


#define RELATION 1
#define OBJECT 1



layer make_region_layer(int batch, int w, int h, int n, int boundary, int classes_act, int classes_obj, int coords_act, int coords_obj)
{
    layer l = {0};
    l.type = REGION;

    l.n = n;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.boundary = boundary;
    l.c = boundary*(classes_act + coords_act + 1) + (n - boundary)*(classes_obj + coords_obj + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes_act = classes_act;
    l.classes_obj = classes_obj;
    l.coords_act = coords_act;
    l.coords_obj = coords_obj;
    l.cost = calloc(1, sizeof(float));
    l.biases = calloc(n*2, sizeof(float));
    l.bias_updates = calloc(n*2, sizeof(float));
    l.outputs = h*w*(boundary*(classes_act + coords_act + 1) + (n - boundary)*(classes_obj + coords_obj + 1));
    l.inputs = l.outputs;
    l.truths = 30*( 4 + 1);
    l.delta = calloc(batch*l.outputs, sizeof(float));
    l.output = calloc(batch*l.outputs, sizeof(float));
    int i;
    for(i = 0; i < n*2; ++i){
        l.biases[i] = .5;
    }

    l.forward = forward_region_layer;
    l.backward = backward_region_layer;
#ifdef GPU
    l.forward_gpu = forward_region_layer_gpu;
    l.backward_gpu = backward_region_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "detection\n");
    srand(0);

    return l;
}

void resize_region_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->outputs = h*w*(l->boundary*(l->classes_act + l->coords_act + 1) + (l->n - l->boundary)*(l->classes_obj + l->coords_obj + 1));
    l->inputs = l->outputs;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta = realloc(l->delta, l->batch*l->outputs*sizeof(float));

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =     cuda_make_array(l->delta, l->batch*l->outputs);
    l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
#endif
}

box get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / w;
    b.y = (j + x[index + 1*stride]) / h;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}

float delta_region_box(box truth, float *x, float *biases, int n, int index, int i, int j, int w, int h, float *delta, float scale, int stride)
{
    box pred = get_region_box(x, biases, n, index, i, j, w, h, stride);
    float iou = box_iou(pred, truth);

    float tx = (truth.x*w - i);
    float ty = (truth.y*h - j);
    float tw = log(truth.w*w / biases[2*n]);
    float th = log(truth.h*h / biases[2*n + 1]);

    delta[index + 0*stride] = scale * (tx - x[index + 0*stride]);
    delta[index + 1*stride] = scale * (ty - x[index + 1*stride]);
    delta[index + 2*stride] = scale * (tw - x[index + 2*stride]);
    delta[index + 3*stride] = scale * (th - x[index + 3*stride]); 
    return iou;
}

void delta_region_class(float *output, float *delta, int index, int class, int classes, tree *hier, float scale, int stride, float *avg_cat)
{
    int i, n;
    if(hier){
        float pred = 1;
        while(class >= 0){
            pred *= output[index + stride*class];
            int g = hier->group[class];
            int offset = hier->group_offset[g];
            for(i = 0; i < hier->group_size[g]; ++i){
                delta[index + stride*(offset + i)] = scale * (0 - output[index + stride*(offset + i)]);
            }
            delta[index + stride*class] = scale * (1 - output[index + stride*class]);

            class = hier->parent[class];
        }
        *avg_cat += pred;
    } else {
        for(n = 0; n < classes; ++n){
            delta[index + stride*n] = scale * (((n == class)?1 : 0) - output[index + stride*n]);
            if(n == class) *avg_cat += output[index + stride*n];
            //if(n == class && delta[index + stride*n] < 0) printf("\noutput[index + stride*n] is: %f\n", output[index + stride*n]);
        }
    }
}

float logit(float x)
{
    return log(x/(1.-x));
}

float tisnan(float x)
{
    return (x != x);
}
//index = entry_index(l, b, n*l.w*l.h, l.coords_obj);
int entry_index(layer l, int batch, int location, int entry)
{
    int n =   location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    if (n < l.boundary){
        return batch*l.outputs + n*l.w*l.h*(l.coords_act + l.classes_act + 1) + entry*l.w*l.h + loc;        
    }
    else{
        return batch*l.outputs + l.boundary*l.w*l.h*(l.coords_act + l.classes_act + 1) + (n - l.boundary)*l.w*l.h*(l.coords_obj + l.classes_obj + 1) + entry*l.w*l.h + loc;
    }
}

void forward_region_layer(const layer l, network net)
{
    int i,j,b,t,n;
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));
    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    if(!net.train) return;
    float avg_iou = 0;
    float recall = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    *(l.cost) = 0;
    for (b = 0; b < l.batch; ++b) {
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
#if RELATION 
                for (n = 0; n < l.boundary; ++n) {
                    int box_index1 = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                    box pred1 = get_region_box(l.output, l.biases, n, box_index1, i, j, l.w, l.h, l.w*l.h);
					int box_index2 = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4);
                    box pred2 = get_region_box(l.output, l.biases, n, box_index2, i, j, l.w, l.h, l.w*l.h);
                    float best_iou = 0;
                    for(t = 0; t < 30; t += 2){
                        box truth1 = float_to_box(net.truth + t*5 + b*l.truths, 1);
						if(!truth1.x || truth1.x > 10000) break;
						box truth2 = float_to_box(net.truth + (t + 1)*5 + b*l.truths, 1);
						if(!truth2.x || truth2.x > 10000) break;
                        float iou1 = box_iou(pred1, truth1);
                        float iou2 = box_iou(pred2, truth2);
						float iou = (iou1 + iou2) / 2;
						//float iou = (iou1);
                        if (iou > best_iou) {
                            best_iou = iou;
                        }
                    }
                    int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 8);
                    avg_anyobj += l.output[obj_index];
                    //printf("\nl.output[obj_index] is:  %f\n", l.output[obj_index]);
                    l.delta[obj_index] = l.noobject_scale * (0 - l.output[obj_index]);
                    if(l.background) l.delta[obj_index] = l.noobject_scale * (1 - l.output[obj_index]);
                    if (best_iou > l.thresh) {
                        l.delta[obj_index] = 0;
                    }
                }
#endif
#if OBJECT
                for (n = l.boundary; n < l.n; ++n) {
                    int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                    box pred = get_region_box(l.output, l.biases, n, box_index, i, j, l.w, l.h, l.w*l.h);
                    float best_iou = 0;
                    for(t = 0; t < 30; t += 1){
                        box truth = float_to_box(net.truth + t*5 + b*l.truths, 1);
                        if(!truth.x) break;
                        if(truth.x > 10000)
                            continue;
                        float iou = box_iou(pred, truth);
                        if (iou > best_iou) {
                            best_iou = iou;
                        }
                    }
                    int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4);
                    avg_anyobj += l.output[obj_index];
                    l.delta[obj_index] = l.noobject_scale * (0 - l.output[obj_index]);
                    if(l.background) l.delta[obj_index] = l.noobject_scale * (1 - l.output[obj_index]);
                    if (best_iou > l.thresh) {
                        l.delta[obj_index] = 0;
                    }

                    if(*(net.seen) < 12800){
                        box truth = {0};
                        truth.x = (i + .5)/l.w;
                        truth.y = (j + .5)/l.h;
                        truth.w = l.biases[2*n]/l.w;
                        truth.h = l.biases[2*n+1]/l.h;
                        delta_region_box(truth, l.output, l.biases, n, box_index, i, j, l.w, l.h, l.delta, .01, l.w*l.h);
                    }
                }
#endif

            }
        }
#if RELATION
//relationship detection
        for(t = 0; t < 30; t += 2){
            box truth1 = float_to_box(net.truth + t*5 + b*l.truths, 1);
			if(!truth1.x || truth1.x > 10000) break;
            box truth2 = float_to_box(net.truth + (t + 1)*5 + b*l.truths, 1);
            //printf("\ntruth2 is: %f, %f, %f, %f\n", truth2.x, truth2.y, truth2.w, truth2.h);
            if(!truth2.x || truth2.x > 10000) break;
            float best_iou = 0;
            int best_n = 0;
            //center is in the middle of 2 box
            //i = ((truth1.x + truth2.x) / 2.0) * l.w;
            //j = ((truth1.y + truth2.y) / 2.0) * l.h;
 
            //center is in first box
			i = (truth1.x ) * l.w;
            j = (truth1.y ) * l.h;

            //center is in second box
			//i = (truth2.x ) * l.w;
            //j = (truth2.y ) * l.h;
                        
			//int u = (truth2.x) * l.w;
			//int v = (truth2.y) * l.h;
            //printf("%d %f %d %f\n", i, truth.x*l.w, j, truth.y*l.h);
            box truth_shift1 = truth1;
			box truth_shift2 = truth2;
            truth_shift1.x = 0;
            truth_shift1.y = 0;
			truth_shift2.x = 0;
            truth_shift2.y = 0;
            //printf("index %d %d\n",i, j);
            for(n = 0; n < l.boundary; ++n){
                int box_index1 = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
				int box_index2 = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4);
                box pred1 = get_region_box(l.output, l.biases, n, box_index1, i, j, l.w, l.h, l.w*l.h);
				box pred2 = get_region_box(l.output, l.biases, n, box_index2, i, j, l.w, l.h, l.w*l.h);
                if(l.bias_match){
                    pred1.w = l.biases[2*n]/l.w;
                    pred1.h = l.biases[2*n+1]/l.h;
					pred2.w = l.biases[2*n]/l.w;
                    pred2.h = l.biases[2*n+1]/l.h;
                }
                //printf("pred: (%f, %f) %f x %f\n", pred.x, pred.y, pred.w, pred.h);
                pred1.x = 0;
                pred1.y = 0;
				pred2.x = 0;
                pred2.y = 0;
                float iou1 = box_iou(pred1, truth_shift1);
                float iou2 = box_iou(pred2, truth_shift2);
				float iou = (iou1 + iou2) / 2;
				//float iou ./darknet detector train cfg/voc.data cfg/yolo-double.cfg darknet19_448.conv.23= (iou1);
                if (iou > best_iou){
                    best_iou = iou;
                    best_n = n;
                }
            }
            //printf("%d %f (%f, %f) %f x %f\n", best_n, best_iou, truth.x, truth.y, truth.w, truth.h);
//i = (truth1.x ) * l.w;
            int box_index1 = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, 0);
			int box_index2 = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, 4);
            float iou1 = delta_region_box(truth1, l.output, l.biases, best_n, box_index1, i, j, l.w, l.h, l.delta, l.coord_scale *  (2 - truth1.w*truth1.h), l.w*l.h);
            float iou2 = delta_region_box(truth2, l.output, l.biases, best_n, box_index2, i, j, l.w, l.h, l.delta, l.coord_scale *  (2 - truth2.w*truth2.h), l.w*l.h);
			float iou = (iou1 + iou2) / 2;
			if(iou > .5) recall += 1;
            avg_iou += iou;
//modity by mengyong            
            if(1){
				//l.delta[best_index + 4] = iou - l.output[best_index + 4];
				int obj_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, l.coords_act);
				avg_obj += l.output[obj_index];
				l.delta[obj_index] = l.object_scale * (1 - l.output[obj_index]);
				if (l.rescore) {
					l.delta[obj_index] = l.object_scale * (iou - l.output[obj_index]);
				}
				if(l.background){
					l.delta[obj_index] = l.object_scale * (0 - l.output[obj_index]);
				}
				int class_act = net.truth[t*(4 + 1) + b*l.truths + 4];
				//printf("2 class is: %d %d\n", class0, class1);
				if(class_act >= l.classes_act) 
					printf("error is: %d", class_act);
				if (l.map) class_act = l.map[class_act];
				int class_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, l.coords_act + 1);
				delta_region_class(l.output, l.delta, class_index, class_act, l.classes_act, l.softmax_tree, l.class_scale, l.w*l.h, &avg_cat);
			}
//modity by mengyong             
            ++count;
            ++class_count;
        }
#endif
#if OBJECT
//object detection
        for(t = 0; t < 30; ++t){
            box truth = float_to_box(net.truth + t*5 + b*l.truths, 1);
            if(!truth.x) break;
            if( truth.x > 10000)
                continue;
            float best_iou = 0;
            int best_n = 0;
            //center is in first box
            i = (truth.x ) * l.w;
            j = (truth.y ) * l.h;
            //printf("%d %f %d %f\n", i, truth.x*l.w, j, truth.y*l.h);
            box truth_shift = truth;
            truth_shift.x = 0;
            truth_shift.y = 0;
            //printf("index %d %d\n",i, j);
            for(n = l.boundary; n < l.n; ++n){
                int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                box pred = get_region_box(l.output, l.biases, n, box_index, i, j, l.w, l.h, l.w*l.h);
                if(l.bias_match){
                    pred.w = l.biases[2*n]/l.w;
                    pred.h = l.biases[2*n+1]/l.h;
                }
                //printf("pred: (%f, %f) %f x %f\n", pred.x, pred.y, pred.w, pred.h);
                pred.x = 0;
                pred.y = 0;
                float iou = box_iou(pred, truth_shift);
                //float iou ./darknet detector train cfg/voc.data cfg/yolo-double.cfg darknet19_448.conv.23= (iou1);
                if (iou > best_iou){
                    best_iou = iou;
                    best_n = n;
                }
            }
            //printf("%d %f (%f, %f) %f x %f\n", best_n, best_iou, truth.x, truth.y, truth.w, truth.h);
//i = (truth1.x ) * l.w;
            int box_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, 0);
            float iou = delta_region_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, l.delta, l.coord_scale *  (2 - truth.w*truth.h), l.w*l.h);
            //float iou = 0;
            if(iou > .5) recall += 1;
            avg_iou += iou;
//modity by mengyong            
            if(1){
                //l.delta[best_index + 4] = iou - l.output[best_index + 4];
                int obj_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, l.coords_obj);
                avg_obj += l.output[obj_index];
                l.delta[obj_index] = l.object_scale * (1 - l.output[obj_index]);
                if (l.rescore) {
                    l.delta[obj_index] = l.object_scale * (iou - l.output[obj_index]);
                }
                if(l.background){
                    l.delta[obj_index] = l.object_scale * (0 - l.output[obj_index]);
                }
                int class = net.truth[t*(4 + 1) + b*l.truths + 4];
                //When labels is in even positions, like 0, 2, etc, the class of the label is person(0).
                if((t % 2) == 0)
                    class = 0;
                //  class = 'person'
                //printf("2 class is: %d %d\n", class0, class1);
                if (l.map) class = l.map[class];
                int class_index = entry_index(l, b, best_n*l.w*l.h + j*l.w + i, l.coords_obj + 1);
                delta_region_class(l.output, l.delta, class_index, class, l.classes_obj, l.softmax_tree, l.class_scale, l.w*l.h, &avg_cat);
            }
//modity by mengyong             
            ++count;
            ++class_count;
        }
#endif
    }
    //printf("\n");
    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    printf("Region Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, Avg Recall: %f,  count: %d\n", avg_iou/count, avg_cat/class_count, avg_obj/count, avg_anyobj/(l.w*l.h*l.n*l.batch), recall/count, count);
}

void backward_region_layer(const layer l, network net)
{
    /*
       int b;
       int size = l.coords + l.classes + 1;
       for (b = 0; b < l.batch*l.n; ++b){
       int index = (b*size + 4)*l.w*l.h;
       gradient_array(l.output + index, l.w*l.h, LOGISTIC, l.delta + index);
       }
       axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
     */
}

void correct_region_boxes(box *boxes, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i){
        box b = boxes[i];
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw); 
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth); 
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        boxes[i] = b;
    }
}


void get_region_boxes_obj(layer l, int w, int h, int netw, int neth, float thresh, float **probs, box *boxes, int only_objectness, int *map, float tree_thresh, int relative)
{

#if OBJECT
    int i,j,n,z;
    float *predictions = l.output;
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = l.boundary; n < l.n; ++n){
            int index = (n - l.boundary)*l.w*l.h + i;
            for(j = 0; j < l.classes_obj; ++j){
                probs[index][j] = 0;
            }
            int obj_index = entry_index(l, 0, n*l.w*l.h + i, 4);
            int box_index = entry_index(l, 0, n*l.w*l.h + i, 0);
            float scale = l.background ? 1 : predictions[obj_index];
            boxes[index] = get_region_box(predictions, l.biases, n, box_index, col, row, l.w, l.h, l.w*l.h);
            int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords_obj + !l.background);
            float max = 0;
            for(j = 0; j < l.classes_obj; ++j){
                int class_index = entry_index(l, 0, n*l.w*l.h + i, 5 + j);
                float prob = scale*predictions[class_index];
                probs[index][j] = (prob > thresh) ? prob : 0;
                //if (j == 1 && prob > thresh) printf("probs[index][j] is: %f\n", probs[index][j]);
                if(prob > max) max = prob;
            }
            probs[index][l.classes_obj] = max;
            if(only_objectness){
                probs[index][0] = scale;
            }
        }
    }
#endif
    correct_region_boxes(boxes, l.w*l.h*(l.n - l.boundary), w, h, netw, neth, relative);
}


void get_region_boxes_act(layer l, int w, int h, int netw, int neth, float thresh, float **probs, box *boxes, int only_objectness, int *map, float tree_thresh, int relative)
{

#if RELATION
    int i,j,n,z;
    float *predictions = l.output;
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.boundary; ++n){
            int index = n*l.w*l.h + i;
            int index1 = index *2;
            int index2 = index * 2 + 1;
            for(j = 0; j < l.classes_act; ++j){
                probs[index][j] = 0;
            }
            int obj_index = entry_index(l, 0, n*l.w*l.h + i, 8);
            int box_index1 = entry_index(l, 0, n*l.w*l.h + i, 0);
            int box_index2 = entry_index(l, 0, n*l.w*l.h + i, 4);
            float scale = l.background ? 1 : predictions[obj_index];
            boxes[index1] = get_region_box(predictions, l.biases, n, box_index1, col, row, l.w, l.h, l.w*l.h);
            boxes[index2] = get_region_box(predictions, l.biases, n, box_index2, col, row, l.w, l.h, l.w*l.h);
            int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords_act + !l.background);
            float max = 0;
            for(j = 0; j < l.classes_act; ++j){
                int class_index = entry_index(l, 0, n*l.w*l.h + i, 9 + j);
                float prob = scale*predictions[class_index];
                probs[index][j] = (prob > thresh) ? prob : 0;
                //if (j == 0 && prob > thresh) printf("probs[index][j] is: %f\n", probs[index][j]);
                if(prob > max) max = prob;
            }
            probs[index][l.classes_act] = max;
            if(only_objectness){
                probs[index][0] = scale;
            }
        }
    }
#endif
    correct_region_boxes(boxes, l.w*l.h*l.boundary*2, w, h, netw, neth, relative);
}

void forward_region_layer_gpu(const layer l, network net)
{
    copy_ongpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
    int b, n;
    for (b = 0; b < l.batch; ++b){
#if RELATION
        for(n = 0; n < l.boundary; ++n){
            int index1 = entry_index(l, b, n*l.w*l.h, 0);
			//int index2 = entry_index(l, b, n*l.w*l.h, 4);
            activate_array_ongpu(l.output_gpu + index1, 2*l.w*l.h, LOGISTIC);
			//activate_array_ongpu(l.output_gpu + index2, 2*l.w*l.h, LOGISTIC);
            int index = entry_index(l, b, n*l.w*l.h, l.coords_act);
            if(!l.background) activate_array_ongpu(l.output_gpu + index,   l.w*l.h, LOGISTIC);
        }
#endif
#if OBJECT
        for(n = l.boundary; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array_ongpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, l.coords_obj);
            if(!l.background) activate_array_ongpu(l.output_gpu + index,   l.w*l.h, LOGISTIC);
        }
#endif
    }
    if (l.softmax) {
#if RELATION
        for(b = 0; b < l.batch; ++b){
            int index_act = entry_index(l, b, 0, l.coords_act + !l.background);
            int batch_offset_act = l.h*l.w*(l.classes_act + l.coords_act + 1);
            softmax_gpu(net.input_gpu + index_act, l.classes_act + l.background, l.boundary, batch_offset_act, l.w*l.h, 1, l.w*l.h, 1, l.output_gpu + index_act);
        }
#endif

#if OBJECT
        for(b = 0; b < l.batch; ++b){
            int index_obj = entry_index(l, b, l.boundary*l.w*l.h, l.coords_obj + !l.background);
            int batch_offset_obj = l.h*l.w*(l.classes_obj + l.coords_obj + 1);
            softmax_gpu(net.input_gpu + index_obj, l.classes_obj + l.background, l.boundary, batch_offset_obj, l.w*l.h, 1, l.w*l.h, 1, l.output_gpu + index_obj);
        }
#endif    
    }

    if(!net.train || l.onlyforward){
        cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        return;
    }

    float *truth_cpu = 0;
    if(net.truth_gpu){
        int num_truth = l.batch*l.truths;
        truth_cpu = calloc(num_truth, sizeof(float));
        cuda_pull_array(net.truth_gpu, truth_cpu, num_truth);
    }
    cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
    forward_region_layer(l, net);
    cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
    if(!net.train) return;
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
}


void backward_region_layer_gpu(const layer l, network net)
{
    int b, n;
    for (b = 0; b < l.batch; ++b){
#if RELATION
        for(n = 0; n < l.boundary; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            gradient_array_ongpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC, l.delta_gpu + index);
			//index = entry_index(l, b, n*l.w*l.h, 4);
			//gradient_array_ongpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC, l.delta_gpu + index);
            index = entry_index(l, b, n*l.w*l.h, l.coords_act);
            if(!l.background) gradient_array_ongpu(l.output_gpu + index,   l.w*l.h, LOGISTIC, l.delta_gpu + index);
        }
#endif
#if OBJECT
        for(n = l.boundary; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            gradient_array_ongpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC, l.delta_gpu + index);
            index = entry_index(l, b, n*l.w*l.h, l.coords_obj);
            if(!l.background) gradient_array_ongpu(l.output_gpu + index,   l.w*l.h, LOGISTIC, l.delta_gpu + index);
        }
#endif
    }
    axpy_ongpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}

void zero_objectness(layer l)
{
    int i, n;
    for (i = 0; i < l.w*l.h; ++i){
        for(n = 0; n < l.n; ++n){
            int obj_index = entry_index(l, 0, n*l.w*l.h + i, 8);
            l.output[obj_index] = 0;
        }
    }
}
