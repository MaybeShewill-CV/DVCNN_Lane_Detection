"""
Set some global config
"""
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
# from config import cfg
cfg = __C

# Train options
__C.TRAIN = edict()

# Set the weighted hat like filter window width
__C.TRAIN.HAT_LIKE_FILTER_WINDOW_WIDTH = 3
# Set the weighted hat like filter window height
__C.TRAIN.HAT_LIKE_FILTER_WINDOW_HEIGHT = 7
# Set the DVCNN training epochs
__C.TRAIN.EPOCHS = 2500
# Set the display step
__C.TRAIN.DISPLAY_STEP = 1
# Set the test display step during training process
__C.TRAIN.TEST_DISPLAY_STEP = 100
# Set the momentum parameter of the optimizer
__C.TRAIN.MOMENTUM = 0.9
# Set the initial learning rate
__C.TRAIN.LEARNING_RATE = 0.001
# Set the GPU resource used during training process
__C.TRAIN.GPU_MEMORY_FRACTION = 0.8
# Set the GPU allow growth parameter during tensorflow training process
__C.TRAIN.TF_ALLOW_GROWTH = False
# Set the DVCNN training batch size
__C.TRAIN.BATCH_SIZE = 64
# Set the DVCNN validation batch size
__C.TRAIN.VAL_BATCH_SIZE = 64




# Test options
__C.TEST = edict()
# Set the weighted hat like filter window width
__C.TEST.HAT_LIKE_FILTER_WINDOW_WIDTH = 3
# Set the weighted hat like filter window height
__C.TEST.HAT_LIKE_FILTER_WINDOW_HEIGHT = 7
