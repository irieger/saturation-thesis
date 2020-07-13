import os
import yaml
import numpy as np


def read_spectra(filename, clip=True):
    """
    Reader for camera response/cie response CSV files.
    Only allowing "classic" format with wl,"r","g","b" for now.
    """
    spectra = np.genfromtxt(filename, delimiter=',')
    if clip:
        spectra[:, 1:4] = np.clip(spectra[:, 1:4], 0, None)
    return spectra

def reduce_spectra(data, low = 0, high = -1, eps = 0.0000000001):
    if high < 0:
        high = data.shape[0] - 1
    
    if low == 0 and high == data.shape[0] - 1:
        for i in range(0, data.shape[0]):
            if np.max(data[i, -3:]) >= eps:
                low = i
                break
        for i in range(high, 0, -1):
            if np.max(data[i, -3:]) >= eps:
                high = i
                break
        
    return (data[low:high], low, high)

def reduce_to_wavelengths(data, min_wl, max_wl):
    try:
        min_idx = np.where(data[:,0] == min_wl)[0][0]
        max_idx = np.where(data[:,0] == max_wl)[0][0]
        return data[min_idx:max_idx + 1, :]
    except:
        return None


def interpolate(spectra):
    """
    Interpolate a given spectra to 1nm step form.
    """
    stepping = int(spectra[1, 0] - spectra[0, 0])
    # Todo: Interpolate to integer steps
    if stepping == 1:
        return spectra
    nsize   = stepping * (spectra.shape[0] - 1) + 1
    upsamp  = np.zeros((nsize, spectra.shape[1]))
    for i in range(nsize):
        if i % stepping == 0:
            upsamp[i, :] = spectra[i//stepping, :]
        else:
            lower = i // stepping
            upper = lower + 1
            w = (stepping - i%stepping) / stepping
            upsamp[i, :] = w * spectra[lower, :] + (1 - w) * spectra[upper, :]
    return upsamp


def read_cie1931():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../external/color-data/observers/cie-1931-2.yaml'))
    f = open(path, 'r')
    data = yaml.load(f)
    cie_data = np.array(data)
    cie_data[:,0] = cie_data[:,0] * 1000000000
    return cie_data

def spectra_integrate(data, response):
    result = np.zeros(data.shape[:-1] + (response.shape[1],))
    for i in range(response.shape[1]):
        result[:, i] = np.sum(np.multiply(data, response[:,i]), axis=-1)
    return result
