import os, sys
import numpy as np
import cv2
import lightning
import torch
import torchvision
from typing import Any, Union


class TrainModule(lightning.LightningModule):
    def __init__(self, configs, *args: Any, **kwargs: Any) -> None:
        '''
        args/kwargs: 不能包含句柄/函数之类的,不然save_hyperparameters报错。
            https://zhuanlan.zhihu.com/p/452780801
        '''
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.configs = configs
    
    def create_model(self):
        ...
    
    def create_loss(self, *args: Any, **kwargs: Any):
        ...


class PredModule():
    def __init__(self, configs) -> None:
        self.configs = configs
    
    def forward(self, data, model=None):
        model_ = self.model if model is None else model
        preds = model_(data)
        return preds
    
    def input_ops(self, data):
        pass
    
    def decode_preds(self, preds):
        pass

    def pred_image(self, image, model=None, params=None):
        ...
    
    def pred_dir(self, img_dir, output_dir, model=None, params=None):
        ...

    def pred_video(self, img_dir, output_dir, model=None, params=None):
        ...

    def get_fps(self, configs, *args: Any, **kwargs: Any):
        ...


class EvalModule():
    def __init__(self, model, configs) -> None:
        self.model = model
        self.configs = configs

    def pred_coco(self, root_dir, set_name='train2017', output_dir='./', save_image=True):
        ...

    def COCOmAP(self, coco_gt, coco_dt, img_ids, iou_type='bbox'):
        ...

    def VOCmAP(self, coco_gt, coco_dt, img_ids, iou_type='bbox'):
        ...


class Model(torch.nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
    
    @staticmethod
    def init_weights(net, init_type='normal', init_gain = 0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and classname.find('Conv') != -1:
                if init_type == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            elif classname.find('BatchNorm2d') != -1:
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)
        print('initialize network weights with %s type' % init_type)
        net.apply(init_func)
        print('initialize network weights with %s type' % init_type)

    def load_checkpoints(self, module, mode='state_dict', device: Union[str, int, list]='cpu', strict=True):
        if isinstance(device, int):
            map_location=lambda storage, loc: storage.cuda(device)
        elif isinstance(device, list):
            map_location={f'cuda:{device[1]}':f'cuda:{device[0]}'}
        else:
            ## 默认cpu
            # map_location=torch.device('cpu')
            map_location=lambda storage, loc: storage
            
        module = torch.load(module, map_location=map_location)
        if 'state_dict' in module.keys():
            model = module['state_dict']
        else:
            model = module
        if list(model.keys())[0].startswith('model.'):
            model = {k[6:]:v for k, v in model.items()}
        self.load_state_dict(model, strict=True)
        return self
    
    def load_pretrained_ckpt(self, model):
        module = torch.load(model)
        if 'state_dict' in module.keys():
            pretrained_dict = module['state_dict']
        else:
            pretrained_dict = module

        model_dict = self.state_dict()
        new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        
    def save_checkpoints(self, name, save_dir='./ckpts', mode=['state_dict']):
        torch.save(self.state_dict(), os.path.join(save_dir, name))
    
    def freeze(self, freeze_layers: Union[str, list, dict]):
        pass
    
    def unfreeze(self, freeze_layers: Union[str, list, dict]=None):
        pass
