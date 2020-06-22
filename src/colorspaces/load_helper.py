import numpy as np
import colour

from .csc import inputColors


cs_mappings = { ### Lower Case!
    'awg': 'ALEXA Wide Gamut',
    'aces-ap0': 'ACES2065-1',
    'p3-d65': 'P3-D65',
    'p3-dci': 'DCI-P3'
}
def convertColor(data, csstring = '', out_whitepoint = (1.0/3, 1.0/3)):
    split_string = csstring.lower().split(':')
    if len(split_string) == 2:
        tc = split_string[0]
        cs = split_string[1]
        if tc == 'srgb':
            tc = colour.RGB_COLOURSPACES['sRGB'].cctf_decoding
        elif tc == 'logc':
            tc = colour.models.log_decoding_ALEXALogC
        elif tc == 'linear':
            tc = None
        
        if cs in cs_mappings:
            cs = cs_mappings[cs]
        
        if not isinstance(tc, str) and len(cs) > 0:
            data = inputColors(data,  color_space=cs, transfer_curve=tc)
        else:
            print('Invalid color space information for ', filename)
            return None
    return data

def importImage(filename, csstring = '', out_whitepoint = (1.0/3, 1.0/3)):
    data = colour.read_image(filename, bit_depth='float32', method='OpenImageIO')[:,:,:3]
    return convertColor(data, csstring, out_whitepoint)