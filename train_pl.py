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
# from torch.optim.swa_utils import AveragedModel, SWALR
import torch.optim as optim
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
# from PIL import ImageDraw, ImageFont, Image
# import colorsys
from pycocotools.coco import COCO

from nets.yolo import YoloBody
from configs.config_v2 import TRAINCONFIGS, EVALCONFIGS
from datasets.coco_dataset_yolo_v5 import COCODetDataset
from loss.yolo_training import YOLOLoss
from interface_v2 import TrainModule


class YOLOTrainModule(TrainModule):
    def __init__(
        self,
        configs=None,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__(configs, *args, **kwargs)
        self.save_hyperparameters()
        self.model = None
        self.model_ema = None
        self.configs = configs
        self.pred_model = None
        self.example_input_array = torch.FloatTensor(np.random.randint(0, 1, size=(1, 3, self.configs.INPUT_SHAPE[0], self.configs.INPUT_SHAPE[1])))

    def init_configs(self):
        return None

    def create_model(self):
        print(f'loading model ...')
        self.model = YoloBody(
            self.configs.ANCHORS_MASK,
            self.configs.NUM_CLASSES,
            self.configs.PHI,
            self.configs.BACKBONE,
            pretrained=self.configs.PRETRAINED,
            input_shape= self.configs.INPUT_SHAPE
        )
        print(f'loading pretrained-model: {self.configs.PRETRAINED} ...')
        self.model.load_checkpoints(module=self.configs.PRETRAINED, device='cpu', strict=True)
        # self.model_ema = ModelEmaV2(self.model)
        # self.model_ema = ModelEMA(self.model)
        print(f'loading pretrained-model: {self.configs.PRETRAINED} complete.')
        print(f'loading model complete.')

        return self.model

    def create_loss(self):
        self.yolo_loss    = YOLOLoss(
            anchors=self.configs.ANCHORS, 
            num_classes=self.configs.NUM_CLASSES, 
            input_shape=self.configs.INPUT_SHAPE, 
            # Cuda, 
            anchors_mask=self.configs.ANCHORS_MASK, 
            label_smoothing=self.configs.LABEL_SMOOTHING
        )
    
    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        images, targets, y_trues = batch[0], batch[1], batch[2]
        batch_size = images.shape[0]
        # with torch.no_grad():
        #     if cuda:
        #         images  = images.cuda(local_rank)
        #         targets = [ann.cuda(local_rank) for ann in targets]
        #         y_trues = [ann.cuda(local_rank) for ann in y_trues]
        ## forward
        outputs         = self.model(images)

        loss_value_all  = 0
        #----------------------#
        #   计算损失
        #----------------------#
        for l in range(len(outputs)):
            loss_item = self.yolo_loss(l, outputs[l], targets, y_trues[l])
            loss_value_all  += loss_item
        loss_value = loss_value_all
        
        # ## ema udpate
        # if self.current_epoch > self.configs.EMA_START_EPOCHS:
        #     # self.model_ema.update_parameters(self.model)
        #     # self.scheduler_ema.step()
        #     self.model_ema.update(self.model)
        
        self.log('train-loss', loss_value.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        
        return loss_value

    def validation_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        images, targets, y_trues = batch[0], batch[1], batch[2]
        batch_size = images.shape[0]
        with torch.no_grad():
            # if cuda:
            #     images  = images.cuda(local_rank)
            #     targets = [ann.cuda(local_rank) for ann in targets]
            #     y_trues = [ann.cuda(local_rank) for ann in y_trues]
            #----------------------#
            #   前向传播
            #----------------------#
            outputs         = self.model(images)

            loss_value_all  = 0
            #----------------------#
            #   计算损失
            #----------------------#
            for l in range(len(outputs)):
                loss_item = self.yolo_loss(l, outputs[l], targets, y_trues[l])
                loss_value_all  += loss_item
            loss_value  = loss_value_all
            self.log('val-loss', loss_value.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
            
            # outputs         = self.model_ema.model(images)
            # loss_value_all_  = 0
            # #----------------------#
            # #   计算损失
            # #----------------------#
            # for l in range(len(outputs)):
            #     loss_item_ = self.yolo_loss(l, outputs[l], targets, y_trues[l])
            #     loss_value_all_  += loss_item_
            # loss_value_  = loss_value_all_
            # self.log('loss', loss_value_.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss_value.item()

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        ## TODO
        ## map进行模型保存依据
        
        ## 统计学习率
        for idx, optim in enumerate(self.optimizer.param_groups):
            self.log(name=f'lr-{idx}', value=optim['lr'], prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
        
        return super().training_epoch_end(outputs)
    
    def test_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        return super().test_step(*args, **kwargs)

    def configure_optimizers(self) -> Any:
        #---------------------------------------#
        #   根据optimizer_type选择优化器
        #---------------------------------------#
        pg0, pg1, pg2 = [], [], []  
        for k, v in self.model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, torch.nn.Parameter):
                pg2.append(v.bias)    
            if isinstance(v, torch.nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)    
            elif hasattr(v, "weight") and isinstance(v.weight, torch.nn.Parameter):
                pg1.append(v.weight)   
        self.optimizer = {
            'adam'  : optim.Adam(pg0, self.configs.INIT_LR, betas = (self.configs.MOMENTUM, 0.999)),
            'sgd'   : optim.SGD(pg0, self.configs.INIT_LR, momentum = self.configs.MOMENTUM, nesterov=True)
        }[self.configs.OPTIMIZER_TYPE]
        self.optimizer.add_param_group({"params": pg1, "weight_decay": self.configs.WEIGHT_DECAY})
        self.optimizer.add_param_group({"params": pg2})
        
        self.scheduler = {
            'cosine': torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.configs.EPOCHS, verbose=True, eta_min=self.configs.INIT_LR),
            'step': torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.configs.LR_STEP_SIZE, gamma=0.8, verbose=True),
            'multistep': torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.configs.LR_MILESTONE, gamma=0.8, verbose=True),
            'exp': torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.8, verbose=True)
        }[self.configs.LR_DECAY_TYPE]
        
        return [self.optimizer], [{'scheduler': self.scheduler}]


class YoloDataModule(L.LightningDataModule):
    def __init__(self, configs) -> None:
        super().__init__()
        self.configs = configs
    
    def train_dataloader(self):
        
        coco = COCO(os.path.join(self.configs['train'].DATASET_ROOT, 'annotations', 'instances_train2017.json'))
        train_dataset = COCODetDataset(
            root_dir=self.configs['train'].DATASET_ROOT, 
            set_name=self.configs['train'].DATASET_SET_NAME, 
            transform=None,
            anchors=self.configs['train'].ANCHORS, anchors_mask=self.configs['train'].ANCHORS_MASK,
            mosaic=self.configs['train'].MOSAIC, mixup=self.configs['train'].MIXUP, 
            mosaic_prob=self.configs['train'].MOSAIC_PROB, mixup_prob=self.configs['train'].MIXUP_PROB, 
            train=True, special_aug_ratio = 0.7, 
            input_shape=self.configs['train'].INPUT_SHAPE,
            coco=coco
        )

        train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        self.train_loader             = DataLoader(
            train_dataset,
            # shuffle = True,
            batch_size = self.configs['train'].BATCH_SIZE,
            num_workers = self.configs['train'].NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
            collate_fn=train_dataset.yolo_dataset_collate,
            sampler=train_sampler
        )
        return self.train_loader
    
    def val_dataloader(self):
        coco = COCO(os.path.join(self.configs['val'].DATASET_ROOT, 'annotations', f'instances_{self.configs["val"].DATASET_SET_NAME}.json'))
        val_dataset = COCODetDataset(
            root_dir=self.configs['val'].DATASET_ROOT, 
            set_name=self.configs['val'].DATASET_SET_NAME, 
            transform=None,
            anchors=self.configs['val'].ANCHORS, anchors_mask=self.configs['val'].ANCHORS_MASK,
            mosaic=self.configs['val'].MOSAIC, mixup=self.configs['val'].MIXUP, 
            mosaic_prob=self.configs['val'].MOSAIC_PROB, mixup_prob=self.configs['val'].MIXUP_PROB, 
            train=True, special_aug_ratio = 0.7, 
            input_shape=self.configs['val'].INPUT_SHAPE,
            coco=coco
        )
        val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        self.val_loader = DataLoader(
            val_dataset,
            # shuffle=False,
            batch_size=self.configs['val'].BATCH_SIZE,
            num_workers=self.configs['val'].NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
            collate_fn=val_dataset.yolo_dataset_collate,
            sampler=val_sampler
        )
        return self.val_loader




if __name__ == '__main__':
    train_module = YOLOTrainModule(configs=TRAINCONFIGS)
    train_module.create_model()
    train_module.create_loss()

    tblogger = TensorBoardLogger(save_dir=TRAINCONFIGS.LOGS_DIR, log_graph=True)
        
    checkpoint_cb = ModelCheckpoint(
        dirpath=tblogger.log_dir,
        filename='{epoch:06d}-{train-loss:.4f}-{val-loss:.4f}',
        save_top_k=3,
        monitor=TRAINCONFIGS.MONITOR,
        mode=TRAINCONFIGS.MONITOR_MODE,
        save_last=True,
        save_on_train_epoch_end=False,
        every_n_epochs=1
    )
    data_module = YoloDataModule(configs={'train': TRAINCONFIGS, 'val': EVALCONFIGS})
    trainer = L.Trainer(
        logger=tblogger,
        
        enable_checkpointing=True,
        callbacks=[checkpoint_cb],
        # default_root_dir: Optional[_PATH] = None,
        # gradient_clip_val: Optional[Union[int, float]] = None,
        # gradient_clip_algorithm: Optional[str] = None,
        num_nodes=1,
        # num_processes: Optional[int] = None,  # TODO: Remove in 2.0
        devices=TRAINCONFIGS.DEVICES,  # 4,
        # gpus: Optional[Union[List[int], str, int]] = None,  # TODO: Remove in 2.0
        # auto_select_gpus: bool = False,
        # tpu_cores: Optional[Union[List[int], str, int]] = None,  # TODO: Remove in 2.0
        # ipus: Optional[int] = None,  # TODO: Remove in 2.0
        enable_progress_bar=True,
        overfit_batches=0.0,
        # track_grad_norm: Union[int, float, str] = -1,
        check_val_every_n_epoch=1,
        fast_dev_run=False,
        accumulate_grad_batches=1,
        max_epochs=TRAINCONFIGS.EPOCHS,
        min_epochs=None,
        max_steps=-1,
        min_steps=None,
        max_time=None,
        # limit_train_batches=100,
        # limit_val_batches=100,
        # limit_test_batches=100,
        # limit_predict_batches=100,
        val_check_interval=1.0,
        log_every_n_steps=10,
        accelerator='gpu',
        strategy = "ddp_find_unused_parameters_false",
        sync_batchnorm=TRAINCONFIGS.SYNC_BN,
        precision=32,
        enable_model_summary=True,
        num_sanity_val_steps=2,
        resume_from_checkpoint=None,
        profiler=None,
        # benchmark: Optional[bool] = None,
        # deterministic: Optional[Union[bool, _LITERAL_WARN]] = None,
        # reload_dataloaders_every_n_epochs: int = 0,
        # auto_lr_find: Union[bool, str] = False,
        # replace_sampler_ddp: bool = True,
        # detect_anomaly: bool = False,
        # auto_scale_batch_size: Union[str, bool] = False,
        # plugins: Optional[Union[PLUGIN_INPUT, List[PLUGIN_INPUT]]] = None,
        # amp_backend: str = "native",
        # amp_level: Optional[str] = None,
        # move_metrics_to_cpu: bool = False,
        # multiple_trainloader_mode: str = "max_size_cycle",
        # inference_mode: bool = True,
    )
    trainer.fit(train_module, datamodule=data_module)

