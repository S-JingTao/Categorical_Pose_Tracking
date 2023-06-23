import copy
import glob
import json
import os

import cv2
import numpy as np
import numpy.ma as ma
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import tqdm as tqdm
from torch.autograd import Variable
import argparse

from lib.models import CASS


class Model(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.casses = self.load_model()

    def load_model(self):
        cass = CASS(self.opt)
        resume_path = os.path.join("models", "cass_best.pth")
        try:
            cass.load_state_dict(torch.load(resume_path), strict=True)
        except:
            raise FileNotFoundError(resume_path)

        return cass

    def get_model(self, cls_idx):
        return self.casses


def to_device(x, cuda=True):
    if cuda:
        return x.cuda()
    else:
        return x.cpu()