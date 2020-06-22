import numpy as np
import colour


def outputColor(xyz, target = 'srgb_oetf', white_point = (1.0/3, 1.0/3)):
    target = target.lower()
    
    if target == 'srgb_oetf':
        return simpleSrgb(xyz, white_point)
    elif target == 'srgb_hermite':
        return hermiteSrgb(xyz, white_point)
    elif target == 'srgb_hermite_rgb':
        return hermiteSrgb(xyz, white_point, overwrite_params={'luma_ratio': 0})

    elif target == 'disp3_oetf':
        return simpleDisplayP3(xyz, white_point)
    elif target == 'disp3_hermite':
        return hermiteDisplayP3(xyz, white_point)
    elif target == 'disp3_hermite_rgb':
        return hermiteDisplayP3(xyz, white_point, overwrite_params={'luma_ratio': 0})

    return None



def simpleSrgb(xyz, white_point = (1.0/3, 1.0/3)):
    """
    Simple (clipping) sRGB conversion
    """

    srgb = colour.RGB_COLOURSPACES['sRGB']
    return colour.XYZ_to_RGB(xyz, np.array(white_point), srgb.whitepoint,
                             srgb.XYZ_to_RGB_matrix, 'CAT02',
                             srgb.encoding_cctf)

def hermiteSrgb(xyz, white_point = (1.0/3, 1.0/3), **kwargs):
    """
    Parameterized sRGB conversion
    """

    srgb = colour.RGB_COLOURSPACES['sRGB']
    return srgb.encoding_cctf(applyHermiteTonemap(xyz, white_point, rgb_space=srgb, **kwargs))


def simpleDisplayP3(xyz, white_point = (1.0/3, 1.0/3)):
    """
    Simple (clipping) DisplayP3 conversion
    """
    
    disp3 = colour.RGB_COLOURSPACES['Display P3']
    return colour.XYZ_to_RGB(xyz, np.array(white_point), disp3.whitepoint,
                             disp3.XYZ_to_RGB_matrix, 'CAT02',
                             disp3.encoding_cctf)

def hermiteDisplayP3(xyz, white_point = (1.0/3, 1.0/3), **kwargs):
    """
    Parameterized Display P3 conversion
    """

    disp3 = colour.RGB_COLOURSPACES['Display P3']
    return disp3.encoding_cctf(applyHermiteTonemap(xyz, white_point, rgb_space=disp3, **kwargs))


spline_curve_defaults = { 'luma_ratio': 1, 'exposure_lin': 1, 'exposure_log': 0, 'offset': 0.00390625, 'maxlog': 8.00002,
                          'minlog': -9, 'maxval': 0.647141, 'spline': [
                              {'c1': -4.51815,     'c2':  3.25685,     'c3': 0.122078, 'c4': 0.0623369, 'knot': 0},
                              {'c1':  1.37784e-14, 'c2': -2.62368e-15, 'c3': 0.9,      'c4': 0.233715,  'knot': 0.258761},
                              {'c1':  1.94884e-14, 'c2': -3.12032e-15, 'c3': 0.9,      'c4': 0.347967,  'knot': 0.385708},
                              {'c1': 10.1545,      'c2': -4.71245,     'c3': 0.9,      'c4': 0.444034,  'knot': 0.492449}
                          ]
                        }
XYZ_TO_YCbCr_Rec2020 = [-1.679915868653143e-06, 1.0000032661742426, -1.532965765091432e-06,     0.009376820215968982, -0.5542542200072184,    0.5007465997790715,        1.164148150852718,  -0.9193503585385239, -0.17181931011016083]
Rec2020_YCbCr_TO_XYZ = [ 0.9504559270516718,    0.2939355109211183, 0.8566310185817273,         1.0,                   3.546704009869884e-06, 1.4144755215108903e-06,    1.0890577507598784,  1.991517837114923,  -0.016039432074386354]


def applyHermiteTonemap(xyz, white_point, spline_params = None, rgb_space = colour.RGB_COLOURSPACES['sRGB'], overwrite_params = {}):
    if spline_params is None:
        spline_params = spline_curve_defaults.copy()
    spline_params.update(overwrite_params)

    XYZ_w  = colour.xy_to_XYZ(np.array(white_point)) * 100
    XYZ_wr = colour.xy_to_XYZ(rgb_space.whitepoint) * 100
    xyz_d65 = colour.adaptation.chromatic_adaptation_VonKries(xyz, XYZ_w, XYZ_wr, transform='CAT02')

    rgb_pure = colour.XYZ_to_RGB(xyz_d65, rgb_space.whitepoint, rgb_space.whitepoint,
                                 rgb_space.XYZ_to_RGB_matrix, None, None)
    rgb_pure = applySplineCurve(rgb_pure, spline_params)

    ycbcr = np.zeros(xyz_d65.shape)
    ycbcr[...,0] = xyz_d65[...,0] * XYZ_TO_YCbCr_Rec2020[0] + xyz_d65[...,1] * XYZ_TO_YCbCr_Rec2020[1] + xyz_d65[...,2] * XYZ_TO_YCbCr_Rec2020[2]
    ycbcr[...,1] = xyz_d65[...,0] * XYZ_TO_YCbCr_Rec2020[3] + xyz_d65[...,1] * XYZ_TO_YCbCr_Rec2020[4] + xyz_d65[...,2] * XYZ_TO_YCbCr_Rec2020[5]
    ycbcr[...,2] = xyz_d65[...,0] * XYZ_TO_YCbCr_Rec2020[6] + xyz_d65[...,1] * XYZ_TO_YCbCr_Rec2020[7] + xyz_d65[...,2] * XYZ_TO_YCbCr_Rec2020[8]
    ycbcr[...,0] = applySplineCurve(ycbcr[...,0], spline_params)
    xyz_d65[...,0] = ycbcr[...,0] * Rec2020_YCbCr_TO_XYZ[0] + ycbcr[...,1] * Rec2020_YCbCr_TO_XYZ[1] + ycbcr[...,2] * Rec2020_YCbCr_TO_XYZ[2]
    xyz_d65[...,1] = ycbcr[...,0] * Rec2020_YCbCr_TO_XYZ[3] + ycbcr[...,1] * Rec2020_YCbCr_TO_XYZ[4] + ycbcr[...,2] * Rec2020_YCbCr_TO_XYZ[5]
    xyz_d65[...,2] = ycbcr[...,0] * Rec2020_YCbCr_TO_XYZ[6] + ycbcr[...,1] * Rec2020_YCbCr_TO_XYZ[7] + ycbcr[...,2] * Rec2020_YCbCr_TO_XYZ[8]
    rgb_luma = colour.XYZ_to_RGB(xyz_d65, rgb_space.whitepoint, rgb_space.whitepoint,
                                 rgb_space.XYZ_to_RGB_matrix, None, None)

    return rgb_pure * (1.0 - spline_params['luma_ratio']) + rgb_luma * spline_params['luma_ratio']
    

def applySplineCurve(vals, spline_params):
    sp = spline_params['spline']

    logvals = np.log2(np.maximum(np.finfo(float).eps, vals * spline_params['exposure_lin'] + spline_params['offset'])) - spline_params['minlog']
    logvals = logvals / (spline_params['maxlog'] - spline_params['minlog']) + spline_params['exposure_log']
    logvals = np.minimum(np.maximum(logvals, sp[0]['knot']), spline_params['maxval'])

    fid = ['c1', 'c2', 'c3', 'c4', 'knot']
    curve_params = np.zeros(vals.shape + (5,))
    for i in range(5):
        curve_params[...,i] = sp[3][fid[i]]

    for i in range(3):
        idx = vals < sp[i]['knot']
        for j in range(len(fid)):
            curve_params[idx, j] = sp[i][fid[j]]
    logvals = logvals - curve_params[...,4]
    pqvals  = logvals * (logvals * (logvals * curve_params[...,0] + curve_params[...,1]) + curve_params[...,2]) + curve_params[...,3]
    return colour.models.eotf_ST2084(pqvals) / 100