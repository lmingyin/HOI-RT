import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

classes = ["sitting", "standing", "use_phone", "phone"]


def get_name_and_suffix(name):
    len_name = len(name)
    if not name[len_name - 1].isalpha():
        name_no_suffix = name[: len_name - 1]
        return name_no_suffix, name[len_name - 1]
    else:
        return name, ''



def convert(size, box):

    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return x, y, w, h


def convert_double_annotation_xml(data_files, data, name, xml_file, face_list, phone_list):
    in_file = open(xml_file)
    out_file = open('%s/%s/labels/%s.txt' % (data_files, data, name), 'w')
    file_name = '%s/%s/labels/%s.txt' % (data_files, data, name)
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    obj_cnt = 0
    for obj in root.iter('object'):
        cls = obj.find('name').text
        act_and_suffix = get_name_and_suffix(cls)
        if act_and_suffix[0] not in face_list:
            continue
        print cls
        phone_index = face_list.index(act_and_suffix[0])
        cls_id_act = phone_index
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        #if w == 0 or h == 0:
        #    print xml_file
        #    raw_input()
        bb_act = convert((w, h), b)
        for obj_phone in root.iter('object'):
            cls = obj_phone.find('name').text
            obj_and_suffix = get_name_and_suffix(cls)
            if obj_and_suffix[0] == phone_list[phone_index] and obj_and_suffix[1] == act_and_suffix[1]:
                obj_cnt += 1
                cls_id_obj = phone_index + 1
                xmlbox = obj_phone.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                #if w == 0 or h == 0:
                #    print xml_file
                #    raw_input()
                bb_obj = convert((w, h), b)
                out_file.write(str(cls_id_act) + " " + " ".join([str(a) for a in bb_act]) + '\n')
                out_file.write(str(cls_id_obj) + " " + " ".join([str(a) for a in bb_obj]) + '\n')
    if obj_cnt == 0:
        os.remove(file_name)
    return obj_cnt

wd = getcwd()

data_sets = ['kick', 'read', 'skateboard', 'ski', 'snowboard', 'surf', 'hold_on_phone', 'work_on_computer']
action_list = ['kick_ball', 'read', 'skateboarding', 'ski', 'snowboarding', 'surfing', 'hold_on_phone', 'work_on_computer']
object_list = ['ball', 'book', 'skateboard', 'skiboard', 'snwboard', 'surfboard', 'phone', 'computer']

data_files = 'RelationDataset'
file = open('trainOurs.txt', 'w')
for data in data_sets:
    if not os.path.exists('%s/%s/labels/' % (data_files, data)):
        os.makedirs('%s/%s/labels/' % (data_files, data))
    img_addr = '%s/%s/JPEGImages' % (data_files, data)
    img_dirs = os.listdir(img_addr)
    length = len(img_dirs)
    img_names = []
    # save all image names without extension
    for image in img_dirs:
        img_name_prefix = os.path.splitext(image)[0]
        xml_file = '%s/%s/Annotations/%s.xml' % (data_files, data, img_name_prefix)
        if os.path.isfile(xml_file):
            # obj_cnt = convert_double_plus_one_annotation_xml(data, img_name_prefix, xml_file, person_list,
            # face_list, phone_list)
            # obj_cnt = convert_triple_plus_one_annotation_xml(data, img_name_prefix, xml_file, person_list,
            #                                                 face_list,  phone_list, hand_list)
            # print 'img_name_prefix is: ', img_name_prefix
            obj_cnt = convert_double_annotation_xml(data_files, data, img_name_prefix, xml_file, action_list, object_list)
            if obj_cnt > 0:
                file.write('%s/%s/%s/JPEGImages/%s\n' % (wd, data_files, data, image))
file.close()
