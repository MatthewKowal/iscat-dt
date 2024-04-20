# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:24:52 2024

install instructions:
    
conda create -n deeptrack python=3.11.7
conda install numpy
pip install deeptrack
pip install tf-keras
pip install opencv-python
pip install spyder
*works so far but i get a lot of warnings when it loads

@author: user1
"""

''' deeptrack training '''
import deeptrack as dt
import matplotlib.pyplot as plt
import numpy as np
import gc
import os
import sys
import skimage.measure
from skimage.filters import gaussian 
import pickle
from skimage import draw
import tensorflow
import tensorflow.keras.layers as layers
import os
from skimage.feature import blob_dog, blob_log, blob_doh


''' backend '''
import os
import time
import numpy as np
import tqdm
# ratiometric particle finder
from collections import deque
# from ultralytics import YOLO
from skimage import color
import random
#particle video
import cv2
import PIL
# #particle list management
from math import dist
#deeptrack particle finder2
from tensorflow.keras.models import load_model
#save bw video
from PIL import ImageFont, ImageDraw
#save color video
import pandas as pd
#load video
import cv2
import numpy as np

''' frontend '''
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import gc
import os
import skimage.measure
from skimage.filters import gaussian 
import pickle
from tensorflow.keras.models import load_model
from skimage.feature import blob_dog, blob_log, blob_doh
import cv2
import imageio
from skimage.transform import downscale_local_mean