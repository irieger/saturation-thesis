import numpy as np
import colour


def YCbCr709(xyz, saturation, white_point):
    wp    = np.array(white_point)
    bt709 = colour.RGB_COLOURSPACES['ITU-R BT.709']
    data  = colour.XYZ_to_RGB(xyz, wp, bt709.whitepoint, bt709.XYZ_to_RGB_matrix, cctf_encoding=None)
    data[data < 0.0] = 0.0
    data = colour.models.rgb.transfer_functions.eotf_inverse_BT1886(data)
    data = colour.RGB_to_YCbCr(data, K=colour.YCBCR_WEIGHTS['ITU-R BT.709']) #, out_legal=False)
    data[...,1] = (data[...,1] - 0.5) * saturation + 0.5
    data[...,2] = (data[...,2] - 0.5) * saturation + 0.5
    data = colour.YCbCr_to_RGB(data, K=colour.YCBCR_WEIGHTS['ITU-R BT.709'])
    data[data < 0.0] = 0.0
    return colour.RGB_to_XYZ(data, bt709.whitepoint, wp, bt709.RGB_to_XYZ_matrix, cctf_decoding=colour.models.rgb.transfer_functions.eotf_BT1886)

def AscCdl(xyz, saturation, white_point):
    wp    = np.array(white_point)
    bt709 = colour.RGB_COLOURSPACES['ITU-R BT.709']
    data  = colour.XYZ_to_RGB(xyz, wp, bt709.whitepoint, bt709.XYZ_to_RGB_matrix, cctf_encoding=None)
    data[data < 0.0] = 0.0
    data = colour.models.rgb.transfer_functions.eotf_inverse_BT1886(data)
    outdata = np.zeros(data.shape)

    lum = 0.2126 * data[...,0] + 0.7152 * data[...,1] + 0.0722 * data[...,1]
    for i in range(3):
        outdata[...,i] = lum + saturation * (data[...,i] - lum)
    data = None
    outdata[outdata < 0] = 0.0
    return colour.RGB_to_XYZ(outdata, bt709.whitepoint, wp, bt709.RGB_to_XYZ_matrix, cctf_decoding=colour.models.rgb.transfer_functions.eotf_BT1886)

def YCbCr2020(xyz, saturation, white_point):
    wp    = np.array(white_point)
    bt2020 = colour.RGB_COLOURSPACES['ITU-R BT.2020']
    data  = colour.XYZ_to_RGB(xyz, wp, bt2020.whitepoint, bt2020.XYZ_to_RGB_matrix, cctf_encoding=None)
    data[data < 0.0] = 0.0
    data = colour.models.rgb.transfer_functions.eotf_inverse_BT1886(data)
    data = colour.RGB_to_YCbCr(data, K=colour.YCBCR_WEIGHTS['ITU-R BT.2020']) #, out_legal=False)
    data[...,1] = (data[...,1] - 0.5) * saturation + 0.5
    data[...,2] = (data[...,2] - 0.5) * saturation + 0.5
    data = colour.YCbCr_to_RGB(data, K=colour.YCBCR_WEIGHTS['ITU-R BT.2020'])
    data[data < 0.0] = 0.0
    return colour.RGB_to_XYZ(data, bt2020.whitepoint, wp, bt2020.RGB_to_XYZ_matrix, cctf_decoding=colour.models.rgb.transfer_functions.eotf_BT1886)


def YCbCr2100Const(xyz, saturation, white_point):
    wp     = np.array(white_point)
    bt2020 = colour.RGB_COLOURSPACES['ITU-R BT.2020']
    data = 100 * colour.XYZ_to_RGB(xyz, wp, bt2020.whitepoint, bt2020.XYZ_to_RGB_matrix, 'CAT02', cctf_encoding=None)
    data = colour.RGB_to_ICTCP(data)
    data[...,1] = saturation * data[...,1]
    data[...,2] = saturation * data[...,2]
    data = colour.ICTCP_to_RGB(data)
    return 0.01 * colour.RGB_to_XYZ(data, bt2020.whitepoint, wp, bt2020.RGB_to_XYZ_matrix, 'CAT02', cctf_encoding=None)


def JzAzBz(xyz, saturation, white_point):
    XYZ_w  = colour.xy_to_XYZ(np.array(white_point)) * 100
    XYZ_wr = colour.xy_to_XYZ(np.array([0.3127,   0.3290])) * 100
    data = colour.adaptation.chromatic_adaptation_VonKries(xyz * 100, XYZ_w, XYZ_wr, transform='CAT02')
    data[data < 0.000000001] = 0.000000001
    data = colour.XYZ_to_JzAzBz(data)

    data[...,1] = saturation * data[...,1]
    data[...,2] = saturation * data[...,2]
    
    data = colour.JzAzBz_to_XYZ(data)
    return colour.adaptation.chromatic_adaptation_VonKries(data, XYZ_wr, XYZ_w, transform='CAT02') * 0.01