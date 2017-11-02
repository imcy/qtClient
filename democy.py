#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
# Add caffe to PYTHONPATH
caffe_path = osp.join('/home/robot/py-faster-rcnn', 'caffe-fast-rcnn', 'python')
add_path(caffe_path)
# Add lib to PYTHONPATH
lib_path = osp.join('/home/robot/py-faster-rcnn', 'lib')
add_path(lib_path)

from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import pandas as pd
from PIL import Image,ImageDraw
import math

CLASSES = ('__background__',
           'box','shoes','cup','spoon','toothbrush','scissors','toothpaste',
	   'pen','tape','razor','bottle','hat','plate','glasses','ball','cuthandle')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}
f = open('cut.txt', 'w')

def getAngle(image,imageall,diff,bbox):
    image = cv2.imread(image)
    imageall = cv2.imread(imageall)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    # blur and threshold the image
    blurred = cv2.blur(gradient, (9, 9))
    (_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)
    (_,cnts,_)= cv2.findContours(thresh.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts,None,cv2.contourArea, reverse=True)[0]
    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))
    width=math.sqrt((box[3][0]-box[0][0])**2+(box[3][1]-box[0][1])**2)
    height=math.sqrt((box[1][0]-box[0][0])**2+(box[1][1]-box[0][1])**2)
    angle=math.acos((box[3][0]-box[0][0])/width)*(180/math.pi)
    for i in box:
       i[0]+=diff[0]
       i[1]+=diff[1]
    # draw a bounding box arounded the detected barcode and display the image
    image=cv2.drawContours(imageall,[box], -1, (0, 255, 0), 3)
 
    if bbox[2]-bbox[0]>bbox[3]-bbox[1]:
	angle+=90
	width=height
    #if width>height:
    #	angle+=90
    #	width=height
    return angle,width

def vis_detections(im, class_name,dets,im_name,classNum,thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    #fig, ax = plt.subplots(figsize=(12, 12))
    #ax.imshow(im, aspect='equal')
    imageall=im_name
    img = Image.open(imageall)
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
	'''       
	ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
	'''
	cut=im[bbox[1]:bbox[3], bbox[0]:bbox[2]]
	cv2.imwrite(dir+'/cut/'+str(i)+"cut.png", cut, params=None)
	
	f.writelines(class_name)
	print class_name
	f.writelines(",")
	imagecut=dircut+'/'+str(i)+"cut.png"
	diff=[bbox[0],bbox[1]]
	angle,width=getAngle(imagecut,imageall,diff,bbox)
	height=33
	
	result=[round((bbox[2]-bbox[0])/2+bbox[0],2),round((bbox[3]-bbox[1])/2+bbox[1],2),round(angle,2),height,round(width,2)]
	print result
	f.write(str(result[0]))
	f.write(",")
	f.write(str(result[1]))
	f.write(",")
	f.write(str(result[2]))
	f.write(",")
	f.write(str(result[4]))
	f.write("\n")

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
	img = Image.open(image_name)
        vis_detections(im, cls, dets, image_name,str(cls_ind),thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))
   
    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print ('\n\nLoaded network {:s}'.format(caffemodel))
    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    global dir 
    global dircut
    dir= '/home/robot/cy/socket/qtproject/images'
    dircut='/home/robot/cy/socket/qtproject/images/cut'
    filelist = []
    global dataAll
    dataAll=[]
    filenames = os.listdir(dir)
    for fn in filenames:
        if fn.endswith('png'):
		fullfilename = os.path.join(dir, fn)
        	filelist.append(fullfilename)
    for im_name in filelist:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_name) 	
    f.close()
	
