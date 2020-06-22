import numpy as np
import colour


def inputColors(data, color_space='DCI-P3', transfer_curve=None, white_point = (1.0/3, 1.0/3)):
    """
    Expects color space names available in colour-science.org.
    Also see availableColorSpaces() for available color spaces

    Only RGB color spaces for now.
    """

    if color_space not in colour.RGB_COLOURSPACES:
        raise Exception('Invalid color space: ' + color_space)

    cs  = colour.RGB_COLOURSPACES[color_space]
    return colour.RGB_to_XYZ(data,
                             cs.whitepoint,
                             np.array(white_point),
                             cs.RGB_to_XYZ_matrix,
                             'CAT02',
                             transfer_curve)


def availableColorSpaces():
    """
    Just give a list of available color spaces
    """
    return [name  for name in colour.RGB_COLOURSPACES]
