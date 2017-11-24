# coding=utf-8
import __init__
import vsrl_utils as vu
import numpy as np
import skimage.io as io
# matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import pylab
from PIL import Image
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
dataType = 'test2014'  # 要转换的COCO数据集的子集名
# labels 目录若不存在，创建labels目录。若存在，则清空目录
if not os.path.exists('%s/labels/test2014/' % dataDir):
    os.makedirs('%s/labels/test2014/' % dataDir)
else:
    shutil.rmtree('%s/labels/test2014/' % dataDir)
    os.makedirs('%s/labels/test2014/' % dataDir)
# filelist 目录若不存在，创建filelist目录。
if not os.path.exists('%s/filelist/' % dataDir):
    os.makedirs('%s/filelist/' % dataDir)
list_file = open('%s/filelist/%s.txt' % (dataDir, dataType), 'w')  # 数据集的图片list保存路径

# Load COCO annotations for V-COCO images
coco = vu.load_coco()



vcoco_anns = ['vcoco_test']
# Load the VCOCO annotations for vcoco_train image set

for anns in vcoco_anns:
    vcoco_all = vu.load_vcoco(anns)
    for x in vcoco_all:
        x = vu.attach_gt_boxes(x, coco)

    # Action classes and roles in V-COCO
    classes = [x['action_name'] for x in vcoco_all]
    # Visualize annotations for the some class

    for cls in classes:
        cls_id = classes.index(cls)
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
            out_file = open('%s/labels/test2014/%s.txt' % (dataDir, file_name[:-4]), 'a')
            if cls_id == classes.index('talk_on_phone'):
                tmp_id = 0
                size = [width, height]
                bb_agent = role_bbox[0]
                bb_object = role_bbox[1]
                out_file.write(str(tmp_id) + " " + " ".join([str(a) for a in bb_agent]) + '\n')
                # out_file.write(str(cls_id + 1) + " " + " ".join([str(a) for a in bb_object]) + '\n')
            out_file.close()
            list_file.write('%s/images/test2014/%s\n' % (dataDir, file_name))
list_file.close()

file_test = open('../looking_phone_test.txt', 'w')
img_addr = ('%s/labels/test2014' % dataDir)
img_dirs = os.listdir(img_addr)
for file_name in img_dirs:
    img_name_prefix = os.path.splitext(file_name)[0]
    file_test.write('%s\n' % img_name_prefix)
file_test.close()
