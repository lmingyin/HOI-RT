# coding=utf-8
import __init__
import vsrl_utils as vu
import numpy as np
import shutil
import os
from os import listdir, getcwd

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
    return x, y, w, h


dataDir = '..'   # COCO数据集所在的路径
dataType = 'trainVCOCO'  # 要转换的COCO数据集的子集名
# labels 目录若不存在，创建labels目录。若存在，则清空目录
if not os.path.exists('%s/labels/train2014/' % dataDir):
    os.makedirs('%s/labels/train2014/' % dataDir)
else:
    shutil.rmtree('%s/labels/train2014/' % dataDir)
    os.makedirs('%s/labels/train2014/' % dataDir)
# filelist 目录若不存在，创建filelist目录。
if not os.path.exists('%s/filelist/' % dataDir):
    os.makedirs('%s/filelist/' % dataDir)
list_file = open('%s/filelist/%s.txt' % (dataDir, dataType), 'w')  # 数据集的图片list保存路径

# Load COCO annotations for V-COCO images
coco = vu.load_coco()

interactions = ['kick', 'read', 'skateboard', 'ski', 'snowboard', 'surf', 'talk_on_phone', 'work_on_computer']
obj_names = ['ball', 'book', 'sktboard', 'skiboard', 'snwboard', 'srfboard', 'phone', 'computer']

vcoco_anns = ['vcoco_trainval']
# Load the VCOCO annotations for vcoco_train image set
wd = getcwd()

for anns in vcoco_anns:
    obj = 0
    for inter in interactions:
        vcoco_all = vu.load_vcoco(anns)
        for x in vcoco_all:
            x = vu.attach_gt_boxes(x, coco)

        # Action classes and roles in V-COCO
        classes = [x['action_name'] for x in vcoco_all]
        if inter not in classes:
            print "error: not the true name!"
            raw_input()
        # Visualize annotations for the some class
        cls_id = classes.index(inter)
        vcoco = vcoco_all[cls_id]
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
            out_file = open('%s/labels/train2014/%s.txt' % (dataDir, file_name[:-4]), 'a')
            size = [width, height]
            bb_agent = convert(size, role_bbox[0])
            bb_object = convert(size, role_bbox[1])
            if bb_object[0] == 0:
                 out_file.close()
                 os.remove('%s/labels/train2014/%s.txt' % (dataDir, file_name[:-4]))
                 continue
            out_file.write(str(obj) + " " + " ".join([str(a) for a in bb_agent]) + '\n')
            out_file.write(str(obj + 1) + " " + " ".join([str(a) for a in bb_object]) + '\n')
            out_file.close()
        obj += 1

img_addr = ('%s/labels/train2014/' % dataDir)
img_dirs = os.listdir(img_addr)
for file_name in img_dirs:
    img_name = os.path.splitext(file_name)[0] + '.jpg'
    list_file.write('%s/%s/images/train2014/%s\n' % (wd, dataDir, img_name))
list_file.close()
