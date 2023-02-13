from yolov5_pl.datasets.coco_dataset_yolo_v5 import COCODetDataset
from torch.utils.data.dataloader import DataLoader
from datasets.dataloader import YoloDataset
import numpy as np
import cv2
import torch
from utils.utils_bbox import DecodeBox









def test_coco_dataset():
    
    from configs.config import CONFIGS
    
    bbox_util = DecodeBox(
        CONFIGS.ANCHORS, 
        80, 
        (CONFIGS.INPUT_SHAPE[0], CONFIGS.INPUT_SHAPE[1]), 
        CONFIGS.ANCHORS_MASK
    )
    
    # root_dir = r'/home/yangliwei/code/PyUtils/PyUtils/test/test_data_format/coco'
    root_dir = r'/home/yangliwei/dataset/coco'
    set_name = 'val2017'

    ds = COCODetDataset(
        root_dir=root_dir, 
        set_name=set_name, 
        transform=None,
        anchors=CONFIGS.ANCHORS, anchors_mask=CONFIGS.ANCHORS_MASK,
        mosaic=CONFIGS.MOSAIC, mixup=CONFIGS.MIXUP, 
        mosaic_prob=CONFIGS.MOSAIC_PROB, mixup_prob=CONFIGS.MIXUP_PROB, 
        train=True, special_aug_ratio = 0.7, 
        input_shape=[640, 640]
    )
    # print(f'{len(ds.classes)}  {ds.classes}')
    # print(f'{len(ds.labels)}  {ds.labels}')
    
    # for img, bboxes, anns in iter(ds):
    #     print(f'{img.shape}  {img.min()}/{img.max()}')
    #     print(f'{len(anns)}: {anns[0].shape}  {anns[0].min()} / {anns[0].max()}')
    #     print(f'{len(anns)}: {anns[1].shape}  {anns[1].min()} / {anns[1].max()}')
    #     print(f'{len(anns)}: {anns[2].shape}  {anns[2].min()} / {anns[2].max()}')
    #     img = img * 128.0 + 127.5
    #     print(f'1  {img.shape=}  {img.min()}  {img.max()}')
    #     img = img.astype(np.uint8)
    #     print(f'2  {img.shape=}  {img.min()}  {img.max()}')
    #     img = img.transpose((1, 2, 0))
    #     print(f'{img.shape=}  {img.min()}  {img.max()}')
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #     # ann1 = np.sum(anns[0], axis=3).transpose((1, 2, 0))
    #     # ann2 = np.sum(anns[1], axis=3).transpose((1, 2, 0))
    #     # ann3 = np.sum(anns[2], axis=3).transpose((1, 2, 0))
    #     # print(f'{ann1.shape}')
    #     # cv2.imwrite('./ann1.jpg', ann1)
    #     # cv2.imwrite('./ann2.jpg', ann2)
    #     # cv2.imwrite('./ann3.jpg', ann3)
    #     cv2.imwrite('./img.jpg', img)
        

    l = DataLoader(
        ds,
        shuffle = False,
        batch_size = 1,
        num_workers = 4,
        pin_memory=True,
        drop_last=True,
        collate_fn=ds.yolo_dataset_collate
    )
    
    for step, (img, bboxes, anns) in enumerate(l):
        print(f'[{step}]  {img.shape}  {img.min()}/{img.max()}')
        print(f'[{step}]  {len(anns)}: {anns[0].shape}  {anns[0].min()} / {anns[0].max()}')
        print(f'[{step}]  {len(anns)}: {anns[1].shape}  {anns[1].min()} / {anns[1].max()}')
        print(f'[{step}]  {len(anns)}: {anns[2].shape}  {anns[2].min()} / {anns[2].max()}')
        img = img.numpy() * 128 + 127.5
        img = img.astype(np.int)
        
        print(f'{type(anns[0])=}  {anns[0].shape}')
        preds = [anns[0][0].permute((0, 3, 1, 2)), anns[1][0].permute((0, 3, 1, 2)), anns[2][0].permute((0, 3, 1, 2))]
        print(f'{type(preds[0])=}  {preds[0].shape}  {preds[1].shape}  {preds[2].shape}')
        # preds = bbox_util.decode_box(preds)
        # #---------------------------------------------------------#
        # #   将预测框进行堆叠，然后进行非极大抑制
        # #---------------------------------------------------------#
        # res = bbox_util.non_max_suppression(
        #     torch.cat(preds, 1), 
        #     CONFIGS.NUM_CLASSES, 
        #     CONFIGS.INPUT_SHAPE, 
        #     image_shape=[640, 640], 
        #     letterbox_image=False, 
        #     conf_thres = CONFIGS.PRED_CONFIDENCE, 
        #     nms_thres = CONFIGS.PRED_NMS_IOU
        # )
        # print(f'{res=}')

        break


def test_yolo_dataset():
    
    from configs.config import CONFIGS
    
    # root_dir = r'/home/yangliwei/code/PyUtils/PyUtils/test/test_data_format/coco'
    root_dir = r'/home/yangliwei/dataset/coco'
    set_name = 'val2017'
    
    with open(CONFIGS.TRAIN_ANNOTATION_PATH, encoding='utf-8') as f:
        train_lines = f.readlines()
    num_train   = len(train_lines)

    ## 因为使用lightning，默认distribute，因此shuffle为False
    shuffle = False

    ds   = YoloDataset(
        train_lines,
        CONFIGS.INPUT_SHAPE,
        CONFIGS.NUM_CLASSES,
        CONFIGS.ANCHORS,
        CONFIGS.ANCHORS_MASK,
        mosaic=CONFIGS.MOSAIC,
        mixup=CONFIGS.MIXUP,
        mosaic_prob=CONFIGS.MOSAIC_PROB,
        mixup_prob=CONFIGS.MIXUP_PROB,
        train=True,
        special_aug_ratio=CONFIGS.SPECIAL_AUG_RATIO
    )


    l = DataLoader(
        ds,
        shuffle = False,
        batch_size = 2,
        num_workers = 4,
        pin_memory=True,
        drop_last=True,
        collate_fn=ds.yolo_dataset_collate
    )
    
    for step, (img, bboxes, anns) in enumerate(l):
        print(f'[{step}]  {img.shape}  {img.min()}/{img.max()}')
        print(f'[{step}]  {len(anns)}: {anns[0].shape}  {anns[0].min()} / {anns[0].max()}')
        print(f'[{step}]  {len(anns)}: {anns[1].shape}  {anns[1].min()} / {anns[1].max()}')
        print(f'[{step}]  {len(anns)}: {anns[2].shape}  {anns[2].min()} / {anns[2].max()}')
        break


    
    
if __name__ == '__main__':
    
    test_coco_dataset()
    # test_yolo_dataset()