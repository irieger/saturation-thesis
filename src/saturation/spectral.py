import os
import numpy as np

from .ksm import KSM
from .meng import Meng


ksm_folder  = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/ksm_dataset_128'))
ksm_calc    = KSM(ksm_folder)
meng_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/meng_dataset_128'))
meng_calc   = Meng.load(meng_folder)


def ksmSaturation(xyz, saturation, luma_preserve = 0.0,
                  white_point = (1.0/3, 1.0/3)):
    """
    Saturate in KSM model
    """
    eps = 0.00000000001
    # Todo White point conversion if not 0.333,0.333
    # ksm_calc   = KSM(load_folder=ksm_folder, uv_grid=True, auto_fill=True)
    data = xyz.copy()
    data[data<eps] = eps
    return ksm_calc.evalImageSaturation(data, saturation, luma_preserve)


def mengSaturation(xyz, saturation, luma_preserve = 0.0,
                   white_point = (1.0/3, 1.0/3)):
    """
    Saturate in Meng2015 model
    """
    eps = 0.00000000001
    # Todo White point conversion if not 0.333,0.333
    data = xyz.copy()
    data[data<eps] = eps
    while len(data.shape) < 3:
        data = data[None,...]
    # Todo line by line to prevent memory problems
    res_spec = np.power(meng_calc.interpolateImage(data, as_xyz=False), saturation)
    res_xyz  = np.zeros(data.shape)
    for y in range(res_xyz.shape[0]):
        for x in range(res_xyz.shape[1]):
            res_xyz[y, x, :] = meng_calc.evalXYZ(res_spec[y,x,:])
    # res_xyz[res_xyz<eps] = eps
    
    y_mult = data[...,1] / res_xyz[...,1]
    return (1 - luma_preserve) * res_xyz + luma_preserve * (res_xyz * np.repeat(y_mult[...,None], 3, axis=-1))