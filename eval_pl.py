#! /usr/bin/env python
# coding: utf-8
import os, sys
import numpy as np
import torch
import torchvision
import torchviz
import lightning as L
from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT, Optional, EPOCH_OUTPUT
from torch.utils.data.dataloader import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
import torch.optim as optim
from lightning.pytorch.callbacks import ModelCheckpoint
from PIL import ImageDraw, ImageFont, Image
import colorsys
import tqdm
from torchvision.transforms import ToTensor
import json

from nets.yolo import YoloBody
from configs.config_v2 import EVALCONFIGS as CONFIGS
from utils.utils_bbox import DecodeBox
from utils.utils import cvtColor, preprocess_input, resize_image
from utils.utils_input import ImageIO, InputOPs
from yolov5_pl.datasets.coco_dataset_yolo_v5 import COCODetDataset


from PyUtils.interface_v2 import EvalModule
from PyUtils.viz.cv_draw import rectangles
from PyUtils.bbox import BBoxes
from PyUtils.viz.pytqdm import redirect_stdout
# from PyUtils.pytorch.coco_data import COCODetDataset

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval






class YoloEvalModule(EvalModule):
    def __init__(self, configs, model, device=0) -> None:
        super().__init__(model=model, configs=configs)
        self.configs = configs
        self.device = device
        self.model = self.load_model(model_file=model, device=device)
        self.bbox_util = DecodeBox(self.configs.ANCHORS, self.configs.NUM_CLASSES, (self.configs.INPUT_SHAPE[0], self.configs.INPUT_SHAPE[1]), self.configs.ANCHORS_MASK)
        self.coco_gt = None
        self.input_shape = configs.INPUT_SHAPE
        self.letterbox_image = configs.LETTERBOX_IMAGE
        self.cuda = configs.CUDA
    
    def load_model(self, model_file, device):
        if isinstance(model_file, str):
            model = YoloBody(
                self.configs.ANCHORS_MASK,
                self.configs.NUM_CLASSES,
                self.configs.PHI,
                self.configs.BACKBONE,
                pretrained=self.configs.PRETRAINED,
                input_shape= self.configs.INPUT_SHAPE
            )
            model.init_weights(model)
            model = model.load_checkpoints(model_file, mode='state_dict', device=device, strict=True)
        else:
            model = model_file
        model.cuda(device)
        model.eval()
        return model

    def pred_image(self, image):  # , class_names, map_out_path
        # f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"), "w", encoding='utf-8') 
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.model(images)
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.configs.NUM_CLASSES, self.configs.INPUT_SHAPE, 
                        image_shape, self.configs.LETTERBOX_IMAGE, conf_thres = self.configs.CONFIDENCE, nms_thres = self.configs.NMS_IOU)
                                                    
            if results[0] is None: 
                return 

            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]

        bboxes = []
        for i, c in list(enumerate(top_label)):
            # predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            # if predicted_class not in class_names:
            #     continue
            bboxes.append([left, top, right-left, bottom-top, c, score])

        return bboxes

    def pred_coco(self, root_dir, set_name='train2017', output_dir='./', save_image=True):
        ann_file = os.path.join(root_dir, 'annotations', 'instances_' + set_name + '.json')
        self.coco_gt = COCO(annotation_file=ann_file)
        img_ids = self.coco_gt.getImgIds()
        
        results = []
        ret_img_ids = []
        tmp_str = ''
        for idx, img_id in enumerate(tqdm.tqdm(img_ids, desc='Predict: ', unit=' img', leave=True, file=sys.stdout, ncols=100, position=0, colour='blue')):
            imgs = self.coco_gt.loadImgs(img_id)
            img_name = imgs[0]['file_name']
            img_file = os.path.join(root_dir, set_name, img_name)
            
            try:
                image = Image.open(img_file)
            except:
                print('Open Error! Try again!')
                continue
            else:
                bboxes = eval_module.pred_image(image=image)
            if bboxes is None: #  or clses is None:
                continue
            ret_img_ids.append(img_id)
            for bbox in bboxes:
                results.append({
                    'image_id': img_id, 
                    'category_id': int(bbox[4]+1), 
                    'score': round(float(bbox[5]), 4), 
                    'bbox': [round(float(bbox[0]), 2), round(float(bbox[1]), 2), round(float(bbox[2]), 2), round(float(bbox[3]), 2)]
                })
                tmp_str += str(img_id) + ' ' + str(int(bbox[4])) + ' ' + str(float(bbox[5])) + ' ' + str(round(float(bbox[0]), 3)) + ' ' + str(round(float(bbox[1]), 3)) + ' ' + str(round(float(bbox[2]), 3)) + ' ' + str(round(float(bbox[3]), 3)) + '\n'
        pred_ret_file = os.path.join(output_dir, f'pre-{set_name}.json')
        with open(pred_ret_file, 'w') as df:
            json.dump(results, df, indent=4)
        img_ids_str = ' '.join([str(img_id) for img_id in img_ids])
        with open(os.path.join(output_dir, f'pre-{set_name}.txt'), 'w') as df:
            df.writelines(img_ids_str)
        
        with open('./tmp_str.txt', 'w') as df:
            df.writelines(tmp_str)
            
        return self.coco_gt, results, ret_img_ids

    def COCOmAP(self, coco_gt, coco_dt, img_ids, iou_type='bbox'):
        coco_eval = COCOeval(cocoGt=coco_gt, cocoDt=coco_dt, iouType='bbox')
        # coco_eval.params.imgIds = img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats

    def test(self):
        gt_ann_file = r'/home/yangliwei/code/yolo/yolov5_pl/val_results/instances_val2017.json'
        dt_ann_file = r'/home/yangliwei/code/yolo/yolov5_pl/val_results/pre-val2017_gt_rand_bbox.json'
        coco_gt = COCO(annotation_file=gt_ann_file)
        coco_dt = coco_gt.loadRes(dt_ann_file)
        coco_eval = COCOeval(cocoGt=coco_gt, cocoDt=coco_dt, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        maxdet_index = coco_eval.params.maxDets.index(100)
        coco_eval.eval['precision'][iou_index, :, :, area_index, maxdet_index].mean()
        
        return coco_eval.stats

if __name__ == '__main__':
    mode = 'test'
    model_file = './model_data/yolov5_s.pth'
    eval_module = YoloEvalModule(configs=CONFIGS, model=model_file, device=0)
    
    if mode == 'coco_map':
        root_dir = r'/home/yangliwei/dataset/coco'
        set_name = 'train2017'
        transformer = [ToTensor()]
        output_dir = '/home/yangliwei/code/yolo/yolov5_pl/val_results'
        gt_ann_file = os.path.join(root_dir, 'annotations', 'instances_' + set_name + '.json')
        dt_ann_file = os.path.join(output_dir, f'pre-{set_name}.json')
        if not os.path.exists(dt_ann_file):
            coco_gt, _, img_ids = eval_module.pred_coco(root_dir=root_dir, set_name=set_name, output_dir=output_dir, save_image=True)
        else:
            coco_gt = COCO(gt_ann_file)
            img_ids = None
            with open(dt_ann_file.replace('json', 'txt'), 'r') as lf:
                img_ids = lf.readlines()[0].strip().split(' ')
            img_ids = [int(img_id) for img_id in img_ids]
                
        coco_pred = coco_gt.loadRes(os.path.join(output_dir, f'pre-{set_name}.json'))
        # run coco eval
        eval_module.COCOmAP(coco_gt=coco_gt, coco_dt=coco_pred, img_ids=img_ids, iou_type='bbox')
    elif mode == 'test':
        eval_module.test()
        