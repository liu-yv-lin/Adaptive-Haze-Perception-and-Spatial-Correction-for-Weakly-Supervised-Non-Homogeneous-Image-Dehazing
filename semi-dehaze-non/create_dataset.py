from __future__ import division
import numpy as np
import sys
import scipy.io as sio
import scipy.ndimage.interpolation
import os
import math
import random
import pdb
import random
import numpy as np
import pickle
import random
import sys
import shutil
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import argparse
from math import log10
import numpy as np
import random
from random import uniform
import h5py
import time
import PIL
from PIL import Image

import h5py
import numpy as np
import matplotlib.pyplot as plt


plt.ion()



index = 1
nyu_depth = h5py.File('/mnt/liuye/dataset/nyu_depth_v2_labeled.mat', 'r')
directory='/mnt/liuye/dataset/train'

if not os.path.exists(directory):
    os.makedirs(directory)
image = nyu_depth['images']
depth = nyu_depth['depths']

img_size = 224

total_num = 0
plt.ion()
for index in range(1448,948,-1):
    gt_image = (image[index, :, :, :]).astype(float)
    gt_image = np.swapaxes(gt_image, 0, 2).astype(float)

    gt_image = gt_image / 255


    gt_depth = depth[index, :, :]
    maxhazy = gt_depth.max()
    minhazy = gt_depth.min()
    gt_depth = (gt_depth) / (maxhazy)

    gt_depth = np.swapaxes(gt_depth, 0, 1)

    gt_depth = scipy.ndimage.zoom(gt_depth, (1, 1), order=1)

    for j in range(4):
        beta = uniform(0.5, 2)
        tx1 = np.exp(-beta * gt_depth)
        a = 1 - 0.5 * uniform(0, 1)
        m = gt_image.shape[0]
        n = gt_image.shape[1]

        rep_atmosphere = np.tile(np.tile(a, [1, 1, 3]), [m, n, 1])
        tx1 = np.reshape(tx1, [m, n, 1])

        max_transmission = np.tile(tx1, [1, 1, 3])

        haze_image = gt_image * max_transmission + rep_atmosphere * (1 - max_transmission)

        total_num = total_num + 1

        h5f=h5py.File('/mnt/liuye/dataset/train/'+str(total_num)+'.h5','w')
        h5f.create_dataset('haze',data=haze_image)
        h5f.create_dataset('trans',data=max_transmission)
        h5f.create_dataset('atom',data=rep_atmosphere)
        h5f.create_dataset('gt',data=gt_image)
        h5f.create_dataset('a', data=a)
        h5f.create_dataset('beta', data=beta)