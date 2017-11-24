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
    return (x, y, w, h)


def draw_bbox(plt, ax, rois, fill=False, linewidth=2, edgecolor=[1.0, 0.0, 0.0], **kwargs):
    for i in range(rois.shape[0]):
        roi = rois[i,:].astype(np.int)
        ax.add_patch(plt.Rectangle((roi[0], roi[1]),
            roi[2] - roi[0], roi[3] - roi[1],
            fill=False, linewidth=linewidth, edgecolor=edgecolor, **kwargs))


def subplot(plt, (Y, X), (sz_y, sz_x) = (10, 10)):
    plt.rcParams['figure.figsize'] = (X*sz_x, Y*sz_y)
    fig, axes = plt.subplots(Y, X)
    return fig, axes

dataDir = os.getcwd()   # COCO数据集所在的路径
dataType = 'train2014'  # 要转换的COCO数据集的子集名
# labels 目录若不存在，创建labels目录。若存在，则清空目录
if not os.path.exists('%s/labels/' % dataDir):
    os.makedirs('%s/labels/' % dataDir)
else:
    shutil.rmtree('%s/labels/' % dataDir)
    os.makedirs('%s/labels/' % dataDir)
# filelist 目录若不存在，创建filelist目录。
if not os.path.exists('%s/filelist/' % dataDir):
    os.makedirs('%s/filelist/' % dataDir)
list_file = open('%s/filelist/%s.txt' % (dataDir, dataType), 'w')  # 数据集的图片list保存路径

# Load COCO annotations for V-COCO images
coco = vu.load_coco()

# Load the VCOCO annotations for vcoco_train image set
vcoco_all = vu.load_vcoco('vcoco_train')
print "length is: ", len(vcoco_all)

for x in vcoco_all:
    x = vu.attach_gt_boxes(x, coco)
    print x

# Action classes and roles in V-COCO
classes = [x['action_name'] for x in vcoco_all]

for i, x in enumerate(vcoco_all):
    print '{:>20s}'.format(x['action_name']), x['role_name']

# Visualize annotations for the some class
cls_id = classes.index('kick')
print 'cls_id is: ', cls_id
vcoco = vcoco_all[cls_id]
print 'vcoco is: ', vcoco
np.random.seed(1)
positive_index = np.where(vcoco['label'] == 1)[0]
positive_index = np.random.permutation(positive_index)

print "positive_index is: ", positive_index
print "length is: ", len(positive_index)
# the demo here laods images from the COCO website,
# you can alternatively use your own local folder of COCO images.
load_coco_image_from_web = False

cc = plt.get_cmap('hsv', lut=4)

for id in positive_index:
    # id = positive_index[i]
    # load image
    coco_image = coco.loadImgs(ids=[vcoco['image_id'][id][0]])[0]
    file_name = coco_image['file_name']
    width = coco_image['width']  # 获取图片尺寸
    height = coco_image['height']  # 获取图片尺寸
    I = io.imread('../images/train2014/%s' % file_name)
    im = np.asarray(I)
    ax = plt.subplot(121)  # use pylab to plot x and y
    ax.imshow(im)  # show the plot on the screen
    sy = 4.
    sx = float(im.shape[1]) / float(im.shape[0]) * sy
    fig, ax = subplot(plt, (1, 1), (sy, sx))
    # ax.set_axis_off()
    # draw bounding box for agent
    draw_bbox(plt, ax, vcoco['bbox'][[id], :], edgecolor=cc(0)[:3])
    print 'vcoco[id]: ', vcoco['bbox'][[id], :]
    role_bbox = vcoco['role_bbox'][id, :] * 1.
    role_bbox = role_bbox.reshape((-1, 4))

    out_file = open('%s/labels/%s.txt' % (dataDir, file_name[:-4]), 'a')
    cls_id = 0
    size = [width, height]
    print 'size is: ', size
    bb_agent = convert(size, role_bbox[0])
    bb_object = convert(size, role_bbox[1])
    out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb_agent]) + '\n')
    out_file.write(str(cls_id + 1) + " " + " ".join([str(a) for a in bb_object]) + '\n')
    out_file.close()

    print 'role_bbox: ', role_bbox
    # print 'include',  vcoco['include'][]
    for j in range(1, len(vcoco['role_name'])):
       if not np.isnan(role_bbox[j, 0]):
           draw_bbox(plt, ax, role_bbox[[j], :], edgecolor=cc(j)[:3])
    ax.imshow(im)
    plt.show()
    #list_file.write('%s/Images/%s\n' % (dataDir, file_name))

list_file.close()