import os
import yaml
import numpy as np


def read_spectra(filename):
    """
    Reader for camera response/cie response CSV files.
    Only allowing classic format with wl,"r","g","b" for now.
    """
    spectra = np.genfromtxt(filename, delimiter=',')
    spectra[:, 1:4] = np.clip(spectra[:, 1:4], 0, None)
    return spectra


def interpolate(spectra):
    """
    Interpolate a given spectra to 1nm step form.
    """
    stepping = int(spectra[1, 0] - spectra[0, 0])
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