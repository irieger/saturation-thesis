import os
import numpy as np

from .ksm import KSM
from .meng import Meng


ksm_folder  = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/ksm_dataset_128'))
ksm_calc    = KSM.load(ksm_folder)
meng_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/meng_dataset_128'))
meng_calc   = Meng.load(meng_folder)


def lumaPreserve(org, saturated, luma_preserve = 0.0):
    if luma_preserve < 0.0 or luma_preserve > 1.0:
        raise Exception('Invalid luma_preserve value')
    new_luma = saturated[...,1].copy()
    idx = new_luma < 0.00000001
    new_luma[idx] = 1.0
    mult      = np.divide(org[...,1], new_luma) * luma_preserve + np.ones(new_luma.shape) * (1 - luma_preserve)
    mult[idx] = 1.0
    return np.multiply(saturated, mult[...,None])


def ksmSaturation(xyz, saturation, luma_preserve = 0.0,
                  white_point = (1.0/3, 1.0/3)):
    """
    Saturate in KSM model
    """
    res = ksm_calc.processImage(saturation, xyz)
    return lumaPreserve(xyz, res, luma_preserve)


def mengSaturation(xyz, saturation, luma_preserve = 0.0,
                   white_point = (1.0/3, 1.0/3)):
    """
    Saturate in Meng2015 model
    """
    res = meng_calc.processImage(saturation, xyz)
    return lumaPreserve(xyz, res, luma_preserve)
