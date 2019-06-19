import os

from easydict import EasyDict as edict

config = edict()

config.TRAIN = edict()
config.TRAIN.batch_size = 10
config.TRAIN.save_interval = 2500
config.TRAIN.log_interval = 10
#config.TRAIN.n_epoch = #10
config.TRAIN.n_step = 600000  # total number of step
config.TRAIN.lr_init = 4e-5  #4e-5 # initial learning rate
config.TRAIN.lr_decay_every_step = 100000  # evey number of step to decay lr
config.TRAIN.lr_decay_factor = 0.333  # decay lr factor
config.TRAIN.weight_decay_factor = 5e-4#5e-4
config.TRAIN.train_mode = 'single'  # single, parallel

config.MODEL = edict()
config.MODEL.model_path = 'models'  # save directory
config.MODEL.model_file = 'pose.npz'  # save file
config.MODEL.n_pos = 19  # number of keypoints + 1 for background
config.MODEL.hin = 368  # input size during training , 240
config.MODEL.win = 368
config.MODEL.hout = int(config.MODEL.hin / 8)  # output size during training (default 46)
config.MODEL.wout = int(config.MODEL.win / 8)
config.MODEL.name = 'hao28_experimental'  # vgg, vggtiny, mobilenet,hao28_experimental

config.MODEL.initial_weights = False  # True,False
config.MODEL.initial_weights_file = 'mobilenet.npz'  # save file


if (config.MODEL.hin % 16 != 0) or (config.MODEL.win % 16 != 0):
    raise Exception("image size should be divided by 16")

config.DATA = edict()
config.DATA.train_data = 'coco'  # coco, custom, coco_and_custom
config.DATA.coco_version = '2017'  # MSCOCO version 2014 or 2017
config.DATA.data_path = 'data'
config.DATA.your_images_path = os.path.join('data', 'your_data', 'images')
config.DATA.your_annos_path = os.path.join('data', 'your_data', 'coco.json')

config.LOG = edict()
config.LOG.vis_path = 'vis'

config.EVAL = edict()
config.EVAL.model = 'pose_best.npz'
config.EVAL.eval_path = 'eval'
config.EVAL.data_idx = -1 # data_idx >= 0 to use specified data
config.EVAL.eval_size = -1 # use first eval_size elements to evaluate, only when data_idx < 0
config.EVAL.plot = False


# config.VALID = edict()

# import json
# def log_config(filename, cfg):
#     with open(filename, 'w') as f:
#         f.write("================================================\n")
#         f.write(json.dumps(cfg, indent=4))
#         f.write("\n================================================\n")
