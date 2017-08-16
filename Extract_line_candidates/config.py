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


# Test options
__C.TEST = edict()
# Set the weighted hat like filter window width
__C.TEST.HAT_LIKE_FILTER_WINDOW_WIDTH = 3
# Set the weighted hat like filter window height
__C.TEST.HAT_LIKE_FILTER_WINDOW_HEIGHT = 7
