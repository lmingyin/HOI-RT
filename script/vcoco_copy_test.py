# coding=utf-8
import __init__
import vsrl_utils as vu
import numpy as np
import shutil
import os


def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[2])/2.0
    y = (box[1] + box[3])/2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    if np.isnan(x):
        x = 0
    if np.isnan(w):
        w = 0
    if np.isnan(y):
        y = 0
    if np.isnan(h):
        h = 0
    return (x, y, w, h)


dataDir = '/home/mengyong/Downloads/coco'   # COCO数据集所在的路径
dataType = 'test'  # 要转换的COCO数据集的子集名
prefType = 'test_pref'

dataDir = '/home/mengyong/Downloads/coco'   # COCO数据集所在的路径
# labels 目录若不存在，创建labels目录。若存在，则清空目录
if not os.path.exists('%s/labels/test/' % dataDir):
    os.makedirs('%s/labels/test/' % dataDir)
else:
    shutil.rmtree('%s/labels/test/' % dataDir)
    os.makedirs('%s/labels/test/' % dataDir)

# filelist 目录若不存在，创建filelist目录。
if not os.path.exists('%s/filelist/' % dataDir):
    os.makedirs('%s/filelist/' % dataDir)
list_file = open('%s/filelist/%s.txt' % (dataDir, dataType), 'w')  # 数据集的图片list保存路径
test_pref_file = open('%s/filelist/%s.txt' % (dataDir, prefType), 'w')  # 数据集的图片list_pref保存路径


# Load COCO annotations for V-COCO images
coco = vu.load_coco()

#vcoco_anns = ['vcoco_trainval']
vcoco_anns = ['vcoco_test']
interactions = ['kick', 'read', 'skateboard', 'ski', 'snowboard', 'surf', 'talk_on_phone', 'work_on_computer']
#interactions = ['talk_on_phone']
# Load the VCOCO annotations for vcoco_train image set

for anns in vcoco_anns:
    vcoco_all = vu.load_vcoco(anns)
    for x in vcoco_all:
        x = vu.attach_gt_boxes(x, coco)
    # Action classes and roles in V-COCO
    classes = [x['action_name'] for x in vcoco_all]
    for cls in interactions:
        cls_id = classes.index(cls)
        # Visualize annotations for the some class
        for cls_sel in interactions:
            vcoco = vcoco_all[classes.index(cls_sel)]
            np.random.seed(1)
            positive_index = np.where(vcoco['label'] == 1)[0]
            positive_index = np.random.permutation(positive_index)
            for id in positive_index:
                coco_image = coco.loadImgs(ids=[vcoco['image_id'][id][0]])[0]
                file_name = coco_image['file_name']
                width = coco_image['width']  # 获取图片尺寸
                height = coco_image['height']  # 获取图片尺寸
                role_bbox = vcoco['role_bbox'][id, :] * 1.
                role_bbox = role_bbox.reshape((-1, 4))
                if not os.path.exists('%s/labels/%s/test' % (dataDir, cls)):
                    os.makedirs('%s/labels/%s/test' % (dataDir, cls))
                out_file = open('%s/labels/%s/test/%s.txt' % (dataDir, cls, file_name[:-4]), 'a')
                if cls == cls_sel:
                    id_index = interactions.index(cls_sel)
                    size = [width, height]
                    bb_agent = role_bbox[0]
                    out_file.write(str(id_index) + " " + " ".join([str(a) for a in bb_agent]) + '\n')
                # if len(role_bbox) > 1:
                #    bb_object = role_bbox[1]
                #    out_file.write(str(cls_id + 1) + " " + " ".join([str(a) for a in bb_object]) + '\n')
                out_file.close()

# create the file of the test images, and the file of the prefix of the test images
img_addr = ('%s/labels/%s/test' % (dataDir, interactions[0]))
img_dirs = os.listdir(img_addr)
for file_name in img_dirs:
    img_name_prefix = os.path.splitext(file_name)[0]
    img_name = img_name_prefix + '.jpg'
    if 'train' in img_name:
        list_file.write('%s/images/train2014/%s\n' % (dataDir, img_name))
    else:
        list_file.write('%s/images/val2014/%s\n' % (dataDir, img_name))
    test_pref_file.write('%s\n' % img_name_prefix)
list_file.close()
test_pref_file.close()




