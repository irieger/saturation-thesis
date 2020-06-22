import numpy as np
import pandas as pd
import os
import yaml

import colour


munsell_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../external/color-data/munsell/real.dat')) 
munsell_data_full = pd.read_csv(munsell_data_path, sep="\s+", skiprows=1, usecols=[0, 1, 2, 3, 4, 5], names=['H', 'V', 'C', 'x', 'y', 'Y'])
munsell_data_np   = munsell_data_full.to_numpy()

def getMunsellData(white_point = (1.0/3, 1.0/3), V = None, H = None):
    target_wp  = np.array(white_point)
    munsell_wp = np.array([0.309, 0.319])
    XYZ_ws  = colour.xy_to_XYZ(munsell_wp) * 100
    XYZ_wt  = colour.xy_to_XYZ(target_wp) * 100

    if V is None:
        V = [4]
    if H is None:
        H = ['5R', '5Y', '10GY', '5BG', '5PB', '5P', '5RP']

    def convert_xyYtoXYZ(data):
        xyz = colour.xyY_to_XYZ(data)
        return colour.adaptation.chromatic_adaptation_VonKries(xyz, XYZ_ws, XYZ_wt, transform='CAT02')
    
    result = []
    for value in V:
        for hue in H:
            idx  = np.logical_and(munsell_data_np[:,0] == hue, munsell_data_np[:,1] == value)
            if np.sum(idx) > 0:
                data = convert_xyYtoXYZ(munsell_data_np[idx, :][:,3:]) / 100
                result.append((data, hue + '_v' + str(value)))
    return result


hung_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../external/color-data/hung-berns/table3.yaml'))

def getHungBernsData(white_point = (1.0/3, 1.0/3)):
    hung_wp = np.array([0.3127,   0.3290])
    XYZ_ws  = colour.xy_to_XYZ(hung_wp) * 100
    XYZ_wt  = colour.xy_to_XYZ(np.array(white_point)) * 100
    def convertWp(data):
        return colour.adaptation.chromatic_adaptation_VonKries(data, XYZ_ws, XYZ_wt, transform='CAT02')

    with open(hung_data_path, 'r') as f:
        ydata = yaml.load(f, Loader=yaml.FullLoader)
        data  = []
        names = []

        for col in ydata:
            names.append(col)
            j = 0
            ldata = np.zeros((4, 3))
            for sat in ydata[col]:
                ldata[j, :] = np.array(ydata[col][sat])
                j += 1
            data.append(convertWp(ldata)/100)
        
        return (data, names)
    
    return None
