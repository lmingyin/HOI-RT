# coding=utf-8

# 该工具用来根据标注文件和原图画出bounding box和类名，来检查标注是否正确

# 使用条件
# 数据集目录结构为：
# Data/worker/class/annotations/xml文件
# Data/worker/class/images/jpg文件
# Data/worker/class/output/即将生成的绘制后的图片
# 该文件需放在Data目录下，worker是进行图片标注的人名，class是物体类名

# 有多少个worker的文件夹，就将多少个文件夹的名字加入sets


import cv2
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import shutil
import numpy as np

sets = ['paul', 'pc']

wd = getcwd()


# 获取worker目录下有多少class目录，返回这些目录的目录名
def get_dirs(sub_dir):
    dirs = []
    dirs_name = os.listdir(wd + '/' + sub_dir)
    for dir_name in dirs_name:
        dirs.append(dir_name)
    return dirs


# 根据标注文件和原图画出bounding box和类名
def draw_box(img_path, ann_path, out_path):
    in_file = open(ann_path)
    input_img = cv2.imread(img_path)
    tree = ET.parse(in_file)
    root = tree.getroot()
    for obj in root.iter('object'):
        cls = obj.find('name').text
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymin').text),
             int(xmlbox.find('ymax').text))
        cv2.rectangle(input_img, (b[0], b[2]), (b[1], b[3]), (0, 0, 255), thickness=2)
        cv2.putText(input_img, cls, (b[0], b[2] + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2)
    cv2.imwrite(out_path, input_img)


# 根据标注文件和原图画出bounding box和类名
def draw_box_vcoco(img_path, cls_txt, bboxs, out_path):
    input_img = cv2.imread(img_path)
    # print 'input_img is: ', input_img
    if bboxs.ndim == 1:
        b = [int(round(bboxs[1])), int(round(bboxs[2])), int(round(bboxs[3])), int(round(bboxs[4]))]
        # print 'b is: ', b
        cv2.rectangle(input_img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), thickness=3)
        #cv2.putText(input_img, cls_txt, (b[0], b[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=3)
    else:
        for box in bboxs:
            # print 'box is: ', box
            b = [int(round(box[1])), int(round(box[2])), int(round(box[3])), int(round(box[4]))]
            # print 'b is: ', b
            cv2.rectangle(input_img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), thickness=3)
            #cv2.putText(input_img, cls_txt, (b[0], b[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=3)
    cv2.imwrite(out_path, input_img)


# 根据标注文件和原图画出bounding box和类名
def draw_box_vcoco_relation(img_path, cls_txt, bboxs, out_path):
    input_img = cv2.imread(img_path)
    # print 'input_img is: ', input_img
    if bboxs.ndim == 1:
        b = [int(round(bboxs[1])), int(round(bboxs[2])), int(round(bboxs[3])), int(round(bboxs[4]))]
        # print 'b is: ', b
        cv2.rectangle(input_img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), thickness=3)
        b = [int(round(bboxs[5])), int(round(bboxs[6])), int(round(bboxs[7])), int(round(bboxs[8]))]
        # print 'b is: ', b
        cv2.rectangle(input_img, (b[0], b[1]), (b[2], b[3]), (255, 0, 255), thickness=3)
        #cv2.putText(input_img, cls_txt, (b[0], b[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=3)
    else:
        for box in bboxs:
            # print 'box is: ', box
            b = [int(round(box[1])), int(round(box[2])), int(round(box[3])), int(round(box[4]))]
            # print 'b is: ', b
            cv2.rectangle(input_img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), thickness=3)
            b = [int(round(box[5])), int(round(box[6])), int(round(box[7])), int(round(box[8]))]
            # print 'b is: ', b
            cv2.rectangle(input_img, (b[0], b[1]), (b[2], b[3]), (255, 0, 255), thickness=3)
            #cv2.putText(input_img, cls_txt, (b[0], b[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=3)
    cv2.imwrite(out_path, input_img)

def copy_img(img_path, out_path):
    input_img = cv2.imread(img_path)
    cv2.imwrite(out_path, input_img)

def IsSubString(SubStrList, Str):
    flag = True
    for substr in SubStrList:
        if not (substr in Str):
            flag = False

    return flag


# 获取FindPath路径下有多少指定格式（FlagStr）的文件，返回所有指定文件的文件名（不加后缀名）
def GetFileList(FindPath, FlagStr=[]):
    import os
    FileList = []
    FileNames = os.listdir(FindPath)
    if (len(FileNames) > 0):
        for fn in FileNames:
            if (len(FlagStr) > 0):
                if (IsSubString(FlagStr, fn)):
                    FileList.append(fn[:-4])
            else:
                FileList.append(fn)

    if (len(FileList) > 0):
        FileList.sort()

    return FileList


dataDir = '/home/mengyong/Downloads/coco'   # COCO数据集所在的路径
interactions = ['kick', 'read', 'skateboard', 'ski', 'snowboard', 'surf', 'talk_on_phone', 'work_on_computer']

for act in interactions:
    label_addr = 'labels/%s/test' % act
    out_addr = 'out/%s' % act
    label_images = os.listdir('%s/%s' % (dataDir, label_addr))
    if not os.path.exists('%s/%s' % (dataDir, out_addr)):
        os.makedirs('%s/%s' % (dataDir, out_addr))
    for label_image in label_images:
        if not os.path.getsize('%s/%s/%s' % (dataDir, label_addr, label_image)):
            continue
        labels = np.loadtxt('%s/%s/%s' % (dataDir, label_addr, label_image))
        # print 'labels is:', labels
        img_name = os.path.splitext(label_image)[0] + '.jpg'
        img_path = ''
        if 'val' in img_name:
            img_path = ('%s/images/val2014/%s' % (dataDir, img_name))
        else:
            img_path = ('%s/images/train2014/%s' % (dataDir, img_name))
        out_path = ('%s/%s/%s' % (dataDir, out_addr, img_name))
        cls_txt = act
        # draw_box_vcoco(img_path, cls_txt, labels, out_path)
        # draw_box_vcoco_relation(img_path, cls_txt, labels, out_path)
        copy_img(img_path, out_path)

'''
for worker in sets:
    dirs = get_dirs(worker)
    for cat in dirs:
        if not os.path.exists('%s/%s/annotations/' % (worker, cat)):
            os.makedirs('%s/%s/annotations/' % (worker, cat))
        if not os.path.exists('%s/%s/output/' % (worker, cat)):
            os.makedirs('%s/%s/output/' % (worker, cat))
        else:
            shutil.rmtree('%s/%s/output/' % (worker, cat))
            os.makedirs('%s/%s/output/' % (worker, cat))
        image_ids = GetFileList(worker + '/' + cat + '/annotations/', ['xml'])
        for image_id in image_ids:
            print(image_id)
            img_path = worker + '/' + cat + '/images/' + image_id + '.jpg'
            ann_path = worker + '/' + cat + '/annotations/' + image_id + '.xml'
            out_path = worker + '/' + cat + '/output/' + image_id + '.jpg'
            # print(img_path)
            # print(out_path)
            draw_box(img_path, ann_path, out_path)
'''