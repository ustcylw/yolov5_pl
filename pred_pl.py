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

import warnings
warnings.filterwarnings("ignore")

from nets.yolo import YoloBody
from configs.config_v2 import PREDCONFIGS as CONFIGS
from utils.utils_bbox import DecodeBox
from utils.utils import cvtColor, preprocess_input, resize_image
from utils.utils_input import ImageIO, InputOPs
from datasets.coco_dataset_yolo_v5 import COCODetDataset

from interface_v2 import PredModule
# from PyUtils.interface import PredModule
from PyUtils.viz.cv_draw import rectangles
from PyUtils.bbox import BBoxes
from PyUtils.viz.pytqdm import redirect_stdout
# from PyUtils.pytorch.coco_data import COCODetDataset

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval




def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)


class YoloPredModule(PredModule):
    def __init__(self, configs, device=0) -> None:
        super().__init__(configs=configs)
        self.configs = configs
        self.device = device
        self.model = self.load_model(model_file=self.configs.MODEL, device=device)
        self.model.eval()
        self.bbox_util = DecodeBox(self.configs.ANCHORS, self.configs.NUM_CLASSES, (self.configs.INPUT_SHAPE[0], self.configs.INPUT_SHAPE[1]), self.configs.ANCHORS_MASK)
        self.coco_gt = None
        
        #---------------------------------------------------------#
        #   图像绘制
        #---------------------------------------------------------#
        hsv_tuples = [(x / self.configs.NUM_CLASSES, 1., 1.) for x in range(self.configs.NUM_CLASSES)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        show_config(**self.configs.__dict__)

    def load_model(self, model_file, device):
        print(f'loading model: {model_file} ...')
        if isinstance(model_file, str):
            model = YoloBody(
                self.configs.ANCHORS_MASK,
                self.configs.NUM_CLASSES,
                self.configs.PHI,
                self.configs.BACKBONE,
                pretrained=self.configs.MODEL,
                input_shape= self.configs.INPUT_SHAPE
            )
            model.init_weights(model)
            model = model.load_checkpoints(model_file, mode='state_dict', device=device, strict=True)
        else:
            model = model_file
        model.eval()
        model.cuda(device)
        print(f'load model {model_file} complete.')
        return model

    def pred_image(self, image, model=None, device=0, params=None, draw=True):
        self.pred_model = self.model
        if isinstance(model, str):
            print(f'loading new model {model} ...')
            self.pred_model = self.load_model(model, device)
            print('{} model, and classes loaded.'.format(model))
        self.pred_model.eval()

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
        image_data  = resize_image(image, (self.configs.INPUT_SHAPE[1], self.configs.INPUT_SHAPE[0]), False)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.device > -1:
                images = images.cuda(f'cuda:{self.device}')
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.pred_model(images)
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(
                torch.cat(outputs, 1), 
                self.configs.NUM_CLASSES, 
                self.configs.INPUT_SHAPE, 
                image_shape, 
                False, 
                conf_thres = self.configs.CONFIDENCE, 
                nms_thres = self.configs.NMS_IOU
            )
                                                    
            if results[0] is None: 
                return image, None, None

            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]
        #---------------------------------------------------------#
        #   设置字体与边框厚度
        #---------------------------------------------------------#
        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.configs.INPUT_SHAPE), 1))
        #---------------------------------------------------------#
        #   计数
        #---------------------------------------------------------#
        if False: # count:
            print("top_label:", top_label)
            classes_nums    = np.zeros([self.configs.NUM_CLASSES])
            for i in range(self.configs.NUM_CLASSES):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.configs.CLASS_NAMES[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        #---------------------------------------------------------#
        #   是否进行目标的裁剪
        #---------------------------------------------------------#
        if False: # crop:
            for i, c in list(enumerate(top_boxes)):
                top, left, bottom, right = top_boxes[i]
                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                right   = min(image.size[0], np.floor(right).astype('int32'))
                
                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)

        bboxes = []
        labels = []
        for i, c in list(enumerate(top_label)):
            predicted_class = self.configs.CLASS_NAMES[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = int(max(0, np.floor(top).astype('int32')))
            left    = int(max(0, np.floor(left).astype('int32')))
            bottom  = int(min(image.size[1], np.floor(bottom).astype('int32')))
            right   = int(min(image.size[0], np.floor(right).astype('int32')))

            # bboxes.append([left, top, right, bottom])
            bboxes.append([left, top, right-left, bottom-top])
            labels.append([predicted_class, int(c), float(score)])
            
            # draw
            if draw:
                label = '{}: {:.2f}'.format(predicted_class, score)
                draw_img = ImageDraw.Draw(image)
                label_size = draw_img.textsize(label, font)
                label = label.encode('utf-8')
                print(label, top, left, bottom, right)
                
                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                for i in range(thickness):
                    draw_img.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
                draw_img.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
                draw_img.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
                del draw_img

        
        # if draw:
        #     image = rectangles(
        #         image=np.array(image), 
        #         bboxes=BBoxes(bboxes, mode='xywh', dtype=int), 
        #         labels=[label[0]+f': {label[2]:.2f}' for label in labels], 
        #         thickness=thickness
        #     )
        # # with redirect_stdout(file=sys.stdout):
        #     # print(f'{bboxes=}  \n{labels}')
        # return Image.fromarray(image), bboxes, labels
        return image, bboxes, labels

    def pred_dir(self, img_dir, output_dir, model=None, params=None):
        img_files = os.listdir(img_dir)
        results = {}
        for idx, img_file in enumerate(tqdm.tqdm(img_files, desc='Predict: ', unit=' img', leave=True, file=sys.stdout, ncols=100, position=0, colour='blue')):
            img_file = os.path.join(img_dir, img_file)
            image = Image.open(img_file)
            ret_img, ret_bboxes, ret_labels = self.pred_image(image=image, model=None, draw=True)
            ret_img.save(os.path.join(output_dir, os.path.basename(img_file)), quality=95, subsampling=0)
    
    def pred_video(self, img_dir, output_dir, model=None, params=None):
        ...

    def get_fps(self, configs, *args: Any, **kwargs: Any):
        ...




if __name__ == '__main__':
    mode = 'pred_dir'
    # model_file = './model_data/yolov5_s.pth'
    model_file = './exps/000001_yolov5/lightning_logs/version_1/epoch=000045-train-loss=0.1896-val-loss=0.1267.ckpt'
    eval_module = YoloPredModule(configs=CONFIGS, device=0)
    

    if mode == 'pred_dir':
        img_dir = r'/home/yangliwei/code/yolo/yolov5_pl/img'
        # img_dir = r'/home/yangliwei/dataset/coco/test2017'
        output_dir = r'/home/yangliwei/code/yolo/yolov5_pl/pred_image/pred_dir'
        eval_module.pred_dir(img_dir=img_dir, output_dir=output_dir)
    elif mode == 'pred_image':
        while True:
            img = input('Input image filename:')
            if img == 'q': break

            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image, bboxes, labels = eval_module.pred_image(image=image, draw=True)
                # r_image.show()
                r_image.save(os.path.join('./pred_image/pred_image', os.path.basename(img)), quality=95, subsampling=0)
    elif mode == 'video':
        pass
    elif mode == 'camera':
        pass
