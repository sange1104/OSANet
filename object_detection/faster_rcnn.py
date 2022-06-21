"""visualize results of the pre-trained model """

# Example
# python demo.py --net res101 --dataset vg --load_dir models --cuda
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from object_detection import _init_paths
import os
import sys
import numpy as np
import argparse
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dset
# from scipy.misc import imread
from imageio import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb

import matplotlib.pyplot as plt

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
    im (ndarray): a color image in BGR order
    Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
              interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

class PretrainedFasterRCNN():
    def __init__(self):
        self.dataset = 'vg'
        self.cfg_file = 'object_detection/cfgs/res101.yml'
        self.net = 'res101'
        self.load_dir = 'object_detection/load_dir' 
        self.classes_dir = 'object_detection/data/genome/1600-400-20'
        self.cuda = True
        self.parallel_type = 0  

        self.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

        cfg_from_file(self.cfg_file)
        cfg_from_list(self.set_cfgs)


        self.lr = cfg.TRAIN.LEARNING_RATE
        self.momentum = cfg.TRAIN.MOMENTUM
        self.weight_decay = cfg.TRAIN.WEIGHT_DECAY


        self.conf_thresh = 0.2
        self.MIN_BOXES = 10
        self.MAX_BOXES = 10


        # Load faster rcnn model
        if not os.path.exists(self.load_dir):
            raise Exception('There is no input directory for loading network from ' + self.load_dir)
        self.load_name = os.path.join(self.load_dir, 'faster_rcnn_{}_{}.pth'.format(self.net, self.dataset))

        # Load classes
        self.classes = ['__background__']
        with open(os.path.join(self.classes_dir, 'objects_vocab.txt')) as f:
            for object in f.readlines():
                self.classes.append(object.split(',')[0].lower().strip())

        # initilize the network here.
        if self.net == 'res101':
            self.class_agnostic = False
            self.fasterRCNN = resnet(self.classes, 101, pretrained=False, class_agnostic=self.class_agnostic)
        else:
            print("network is not defined")
            pdb.set_trace()

        # Build faster rcnn
        self.fasterRCNN.create_architecture()
 
        if self.cuda > 0:
            checkpoint = torch.load(self.load_name)
        else:
            checkpoint = torch.load(self.load_name, map_location=(lambda storage, loc: storage))
        self.fasterRCNN.load_state_dict(checkpoint['model'])
        if 'pooling_mode' in checkpoint.keys():
            POOLING_MODE = checkpoint['pooling_mode'] 
    
    def _initialize(self):

        # initilize the tensor holder here.
        self.im_data = torch.FloatTensor(1)
        self.im_info = torch.FloatTensor(1)
        self.num_boxes = torch.LongTensor(1)
        self.gt_boxes = torch.FloatTensor(1)

        # ship to cuda
        if self.cuda > 0:
            self.im_data = self.im_data.cuda()
            self.im_info = self.im_info.cuda()
            self.num_boxes = self.num_boxes.cuda()
            self.gt_boxes = self.gt_boxes.cuda()

        # make variable
        with torch.no_grad():
            self.im_data = Variable(self.im_data)
            self.im_info = Variable(self.im_info)
            self.num_boxes = Variable(self.num_boxes)
            self.gt_boxes = Variable(self.gt_boxes)

        if self.cuda > 0:
            cfg.CUDA = True
            self.fasterRCNN.cuda() 

        self.fasterRCNN.eval()
        
    def detect_object(self, img):  
        self._initialize()
        start = time.time()
        max_per_image = 100
        thresh = 0.05
        vis = False
 
        # input
        im_in = np.array(img)
        if len(im_in.shape) == 2:
            im_in = im_in[:,:,np.newaxis]
            im_in = np.concatenate((im_in,im_in,im_in), axis=2)
            # rgb -> bgr
        im = im_in[:,:,::-1]

        blobs, im_scales = _get_image_blob(im)
        assert len(im_scales) == 1, "Only single-image batch implemented"
        im_blob = blobs
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        with torch.no_grad():
            self.im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
            self.im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
            self.gt_boxes.resize_(1, 1, 5).zero_()
            self.num_boxes.resize_(1).zero_()
        # pdb.set_trace()
        det_tic = time.time()

        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = self.fasterRCNN(self.im_data, self.im_info, self.gt_boxes, self.num_boxes)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if self.class_agnostic:
                    if cuda > 0:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                   + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                   + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    if self.cuda > 0:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                   + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                   + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                    box_deltas = box_deltas.view(1, -1, 4 * len(self.classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, self.im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1])) 
        pred_boxes /= im_scales[0]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()

        max_conf = torch.zeros((pred_boxes.shape[0]))
        if self.cuda > 0:
            max_conf = max_conf.cuda()
 
        for j in xrange(1, len(self.classes)):
            inds = torch.nonzero(scores[:,j]>thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:,j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if self.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                # keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
                keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                index = inds[order[keep]]
                max_conf[index] = torch.where(scores[index, j] > max_conf[index], scores[index, j], max_conf[index]) 


        if self.cuda > 0:
            keep_boxes = torch.where(max_conf >= self.conf_thresh, max_conf, torch.tensor(0.0).cuda())
        else:
            keep_boxes = torch.where(max_conf >= self.conf_thresh, max_conf, torch.tensor(0.0))
            keep_boxes = torch.squeeze(torch.nonzero(keep_boxes))
        if len(keep_boxes) < self.MIN_BOXES:
            keep_boxes = torch.argsort(max_conf, descending = True)[:self.MIN_BOXES]
        elif len(keep_boxes) > self.MAX_BOXES:
            keep_boxes = torch.argsort(max_conf, descending = True)[:self.MAX_BOXES]
 
        boxes = pred_boxes[keep_boxes]
        objects = torch.argmax(scores[keep_boxes][:,1:], dim=1)
        object_classes = []
        for i in range(len(keep_boxes)):
        #     print('score', scores[i])
            kind = objects[i]+1
            bbox = boxes[i, kind * 4: (kind + 1) * 4]
            # bbox = boxes[i]
            if bbox[0] == 0:
                bbox[0] = 1
            if bbox[1] == 0:
                bbox[1] = 1
            cls = self.classes[objects[i]+1]
            object_classes.append(cls) 
        return object_classes
