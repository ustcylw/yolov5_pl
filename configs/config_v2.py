import os, sys
from utils.utils import get_anchors, get_classes
import time


class BASECONFIGS():
    ## envs
    MASTER_ADDR = '127.0.0.1'
    MASTER_PORT = '29500'
    NCCL_DEBUG = "INFO"
    WORLD_SIZE = 2
    
    ## exps
    TIME = time.strftime("%Y:%m:%d-%H:%M:%S")
    EXP_NUM = '000001'
    PRE_SYMBOL = 'yolov5'
    ROOT_DIR = './'
    EXP_DIR = os.path.join(ROOT_DIR, 'exps', f'{EXP_NUM}_{PRE_SYMBOL}')
    LOGS_DIR = EXP_DIR
    CKPTS_DIR = os.path.join(ROOT_DIR, 'exps', f'{EXP_NUM}_{PRE_SYMBOL}', TIME, 'ckpts')
    


class TRAINCONFIGS(BASECONFIGS):

    if not os.path.exists(BASECONFIGS.EXP_DIR):
        os.makedirs(BASECONFIGS.EXP_DIR)

    ## train
    START_EPOCHS = 0
    EPOCHS = 100
    SYNC_BN = False
    PRECISION = False
    SAVE_PERIOD         = 10
    EVAL_PERIOD         = 10
    NUM_WORKERS         = 4
    EMA                 = True
    EMA_STEPS           = 16
    EMA_DECAY           = 0.99998
    EMA_START_EPOCHS    = int(EPOCHS * 0.8 + START_EPOCHS)
    BATCH_SIZE    = 32
    DEVICES = [5, 6, 7, 8, 9]
    
    # dataset
    DATASET_ROOT = r'/home/yangliwei/dataset/coco'
    DATASET_SET_NAME = 'train2017'
    
    
    ## task
    CLASSES_PATH    = 'model_data/coco_classes.txt'
    ANCHORS_PATH    = 'model_data/yolo_anchors.txt'
    ANCHORS_MASK    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    CLASS_NAMES, NUM_CLASSES = get_classes(CLASSES_PATH)
    ANCHORS, NUM_ANCHORS     = get_anchors(ANCHORS_PATH)
    INPUT_SHAPE     = [640, 640]
    BACKBONE        = 'cspdarknet'
    PRETRAINED = f'./exps/000001_yolov5/lightning_logs/version_1/epoch=000096-train-loss=0.1729-val-loss=0.1204.ckpt'
    PHI             = 's'
    MOSAIC              = False
    MOSAIC_PROB         = 0.5
    MIXUP               = False
    MIXUP_PROB          = 0.5
    SPECIAL_AUG_RATIO   = 0.7
    LABEL_SMOOTHING     = 0.006
    OPTIMIZER_TYPE      = "adam"
    MOMENTUM            = 0.937
    WEIGHT_DECAY        = 5e-4
    INIT_LR             = 1e-4
    MIN_LR              = INIT_LR * 0.01
    LR_DECAY_TYPE       = "step"
    LR_STEP_SIZE        = 3
    LR_MILESTONE  = [20, 80, 120, 150]

    MONITOR = 'val-loss'
    MONITOR_MODE = 'min'
    
    def to_string():
        return ''



class PREDCONFIGS(BASECONFIGS):
    
    ## task
    CLASSES_PATH    = 'model_data/coco_classes.txt'
    ANCHORS_PATH    = 'model_data/yolo_anchors.txt'
    ANCHORS_MASK    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    CLASS_NAMES, NUM_CLASSES = get_classes(CLASSES_PATH)
    ANCHORS, NUM_ANCHORS     = get_anchors(ANCHORS_PATH)
    INPUT_SHAPE     = [640, 640]
    BACKBONE        = 'cspdarknet'
    PHI             = 's'

    ## pred
    MODEL = './exps/000001_yolov5/lightning_logs/version_4/epoch=000007-train-loss=0.1330-val-loss=0.1314.ckpt'
    CONFIDENCE = 0.5
    #---------------------------------------------------------------------#
    #   非极大抑制所用到的nms_iou大小
    #---------------------------------------------------------------------#
    NMS_IOU = 0.5
    #---------------------------------------------------------------------#
    #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
    #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
    #---------------------------------------------------------------------#
    LETTERBOX_IMAGE = True
    CUDA = True
        
    def to_string():
        return ''



class EVALCONFIGS(BASECONFIGS):
    
    ## eval
    CONFIDENCE = 0.001
    NMS_IOU = 0.5
    
    
    ## task
    CLASSES_PATH    = 'model_data/coco_classes.txt'
    ANCHORS_PATH    = 'model_data/yolo_anchors.txt'
    ANCHORS_MASK    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    CLASS_NAMES, NUM_CLASSES = get_classes(CLASSES_PATH)
    ANCHORS, NUM_ANCHORS     = get_anchors(ANCHORS_PATH)
    INPUT_SHAPE     = [640, 640]
    BACKBONE        = 'cspdarknet'
    PHI             = 's'
    MOSAIC              = False
    MOSAIC_PROB         = 0.5
    MIXUP               = False
    MIXUP_PROB          = 0.5
    SPECIAL_AUG_RATIO   = 0.7
    PRETRAINED = None
    BATCH_SIZE    = 32
    NUM_WORKERS         = 4

    LETTERBOX_IMAGE = False
    CUDA = True
    
    # dataset
    DATASET_ROOT = r'/home/yangliwei/dataset/coco'
    DATASET_SET_NAME = 'val2017'

    def to_string():
        return ''

