"""
Some global parameters, you should not change it easily
"""


DVCNN_ARCHITECTURE = {
    'conv1': {
        'ksize': [5, 5, 3, 16],
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'trainable': True
    },
    'relu1': {
        'method': 'RELU'
        },
    'pool1': {
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': 'SAME'
    },
    'conv2_front': {
        'ksize': [5, 5, 16, 16],
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'trainable': True
    },
    'conv2_top': {
        'ksize': [5, 5, 3, 16],
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'trainable': True
    },
    'relu2': {
        'method': 'RELU'
    },
    'pool2': {
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': 'SAME'
    },
    'conv3': {
        'ksize': [5, 5, 16, 32],
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'trainable': True
    },
    'relu3': {
        'method': 'RELU'
    },
    'pool3': {
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': 'SAME'
    },
    'conv4': {
        'ksize': [5, 5, 32, 32],
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'trainable': True
    },
    'relu4': {
        'method': 'RELU'
    },
    'pool4': {
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': 'SAME'
    },
    'conv5': {
        'ksize': [5, 5, 32, 64],
        'strides': [1, 1, 1, 1],
        'padding': 'SAME',
        'trainable': True
    },
    'relu5': {
        'method': 'RELU'
    },
    'pool5': {
        'ksize': [1, 2, 2, 1],
        'strides': [1, 2, 2, 1],
        'padding': 'SAME'
    },
    'fc6': {
        'ksize': [4, 4, 64, 256],
        'strides': [1, 4, 4, 1],
        'padding': 'VALID',
        'trainable': True
    },
    'relu6': {
        'method': 'RELU'
    },
    'concat7': {
        'axis': 3
    },
    'fc8': {
        'ksize': [1, 1, 512, 2],
        'strides': [1, 1, 1, 1],
        'padding': 'VALID',
        'trainable': True
    }
}
