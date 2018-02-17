
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random

from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt
import numpy as np
import IPython

from encoding import DecodingNet



# Test using either torch or numpy/image transforms and Imagenet