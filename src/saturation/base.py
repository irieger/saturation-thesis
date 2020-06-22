from .regular import *
from .spectral import *

def saturate(xyz, saturation, model = 'bt709', white_point = (1.0/3, 1.0/3),
             **kwargs):
    """
    Saturate image data
    """

    model = model.lower()

    if model == 'bt709':
        return YCbCr709(xyz, saturation, white_point)
    elif model == 'asccdl':
        return AscCdl(xyz, saturation, white_point)
    elif model == 'bt2020':
        return YCbCr2020(xyz, saturation, white_point)
    elif model == 'bt2100const':
        return YCbCr2100Const(xyz, saturation, white_point)
    elif model == 'jzazbz':
        return JzAzBz(xyz, saturation, white_point)

    elif model == 'ksm':
        return ksmSaturation(xyz, saturation, white_point=white_point,
                             **kwargs)
    elif model == 'ksm0.00':
        return ksmSaturation(xyz, saturation, white_point=white_point,
                             luma_preserve=0.0)
    elif model == 'ksm0.25':
        return ksmSaturation(xyz, saturation, white_point=white_point,
                             luma_preserve=0.25)
    elif model == 'ksm0.50':
        return ksmSaturation(xyz, saturation, white_point=white_point,
                             luma_preserve=0.5)
    elif model == 'ksm0.75':
        return ksmSaturation(xyz, saturation, white_point=white_point,
                             luma_preserve=0.75)
    elif model == 'ksm1.00':
        return ksmSaturation(xyz, saturation, white_point=white_point,
                             luma_preserve=1.0)

    elif model == 'meng':
        return mengSaturation(xyz, saturation, white_point=white_point,
                              **kwargs)
    elif model == 'meng0.00':
        return mengSaturation(xyz, saturation, white_point=white_point,
                              luma_preserve=0.0)
    elif model == 'meng0.25':
        return mengSaturation(xyz, saturation, white_point=white_point,
                              luma_preserve=0.25)
    elif model == 'meng0.50':
        return mengSaturation(xyz, saturation, white_point=white_point,
                              luma_preserve=0.5)
    elif model == 'meng0.75':
        return mengSaturation(xyz, saturation, white_point=white_point,
                              luma_preserve=0.75)
    elif model == 'meng1.00':
        return mengSaturation(xyz, saturation, white_point=white_point,
                              luma_preserve=1.0)

    else:
        raise Exception('Requested model is not implemented')