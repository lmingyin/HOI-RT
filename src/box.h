#ifndef BOX_H
#define BOX_H
#include "darknet.h"

typedef struct{
    float dx, dy, dw, dh;
} dbox;

box float_to_box(float *f, int stride);
float box_iou(box a, box b);
float box_rmse(box a, box b);
dbox diou(box a, box b);
void do_nms(box *boxes, float **probs, int total, int classes, float thresh);
void do_nms_sort(box *boxes, float **probs, int total, int classes, float thresh);
void do_nms_sort_act_valid(box *boxes, float **probs, int total, int classes, float thresh);
void do_nms_obj(box *boxes, float **probs, int total, int classes, float thresh);
void do_nms_act(box *boxes, float **probs, int total, int classes, float thresh);
box decode_box(box b, box anchor);
box encode_box(box b, box anchor);

float min_float(float a, float b);
float max_float(float a, float b);
float caculate_prob_dis(box obj1, box obj2);
void merge_act_and_obj(box *boxes_act, float **probs_act, int num_act, int classes_act,
                       box *boxes_obj, float **probs_obj, int num_obj, int classes_obj,
                       float thresh);

#endif
