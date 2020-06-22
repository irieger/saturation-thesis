import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
import pandas as pd
import os

import colour
import colour.plotting

from .reference_data import *
#import .reference_data as reference_data


colour.plotting.colour_style()


def createHull(resolution=10, transfer_curve=None):
    output = np.zeros((resolution ** 3, 3))
    divisor = 1.0 / (resolution - 1)
    for r in range(resolution):
        for g in range(resolution):
            for b in range(resolution):
                idx = r * resolution * resolution + g * resolution + b
                output[idx, 0] = r * divisor
                output[idx, 1] = g * divisor
                output[idx, 2] = b * divisor
    if not transfer_curve is None:
        output = transfer_curve(output)
    return output

bt2020 = colour.RGB_COLOURSPACES['ITU-R BT.2020']
itur_bt709_hull = createHull(transfer_curve=lambda x: colour.RGB_to_YCbCr(x, out_legal=False))
ictcp_hull      = createHull(transfer_curve=lambda x: colour.RGB_to_ICTCP(colour.models.rgb.transfer_functions.st_2084.eotf_ST2084(x * colour.models.rgb.transfer_functions.st_2084.eotf_inverse_ST2084(1000))))
jzazbz_hull     = createHull(transfer_curve=lambda x: colour.XYZ_to_JzAzBz(colour.RGB_to_XYZ(colour.models.rgb.transfer_functions.st_2084.eotf_ST2084(x * colour.models.rgb.transfer_functions.st_2084.eotf_inverse_ST2084(1000)),
                             bt2020.whitepoint,
                             bt2020.whitepoint,
                             bt2020.RGB_to_XYZ_matrix,
                             None,
                             None)))

colors            = ['r', 'g', 'b', 'c', 'm', 'y']
line_style        = ['--', '-.']
line_point_marker = '+'
scatter_marker    = ['o', '*', 'x']
hue_line_color    = '0.65'
hue_line_style    = ':'
hue_point_style   = '.'


def lineStyle(cnt):
    # line_point_marker not shown!
    c = cnt % len(colors)
    s = (cnt // len(colors)) % len(line_style)
    return (colors[c], line_point_marker, line_style[s])
    # return colors[c] + line_point_marker + line_style[s]

def scatterStyle(cnt):
    c = cnt % len(colors)
    m = (cnt // len(colors)) % len(scatter_marker)
    return (colors[c], scatter_marker[m])

def plotHull(data, ax, alpha=0.18):
    hull = ConvexHull(data)
    poly = plt.Polygon(data[hull.vertices,:], ec='black', fc='gray', alpha=alpha)

    ax.add_patch(poly)



def plot_uv(lines = [], scatter = [], hue_lines = [], white_point = (1.0/3, 1.0/3), show_diagram_colours=False,
            filename = None, save_only = False):
    """
    Plot uv chromaticities in the CIE1976UCS space. lines and scatters data prepared
    as in plotAll
    """

    # TODO: Background color

    colour.plotting.plot_RGB_colourspaces_in_chromaticity_diagram_CIE1976UCS(colourspaces=['sRGB', 'DCI-P3'], standalone=False,
                        show_whitepoints = False, show_diagram_colours=show_diagram_colours)

    if not white_point is None:
        wp_uv = colour.xy_to_Luv_uv(np.array(white_point))
        plt.scatter(wp_uv[0], wp_uv[1], c='k')
    
    for data,label in hue_lines:
        uv = colour.xy_to_Luv_uv(data)
        plt.plot(uv[...,0], uv[...,1], label=label, color=hue_line_color, linestyle=hue_line_style, marker=hue_point_style)

    cnt = 0
    for data,label in lines:
        uv = colour.xy_to_Luv_uv(data)
        lstyle = lineStyle(cnt)
        plt.plot(uv[...,0], uv[...,1], label=label, color=lstyle[0], linestyle=lstyle[2], marker=lstyle[1])
        cnt += 1
    for data,label in scatter:
        uv = colour.xy_to_Luv_uv(data)
        c,m = scatterStyle(cnt)
        plt.scatter(uv[...,0], uv[...,1], label=label, c=c, marker=m)
        cnt += 1

    plt.xlim(-0.05, 0.65)
    plt.ylim(-0.03, 0.65)

    if filename:
        filename = filename.format('uv')
    fig = colour.plotting.render(standalone=True,
                                 x_tighten=True,
                                 y_tighten=True,
                                 filename=filename,
                                 transparent_background=False)
    # if filename and not save_only:
    #     print('plot called')
    #     plt.show()


def plot_xy(lines = [], scatter = [], hue_lines = [], white_point = (1.0/3, 1.0/3),
            filename = None, save_only = False):
    """
    Plot xy chromaticities in the CIE1931 chromaticity diagram. lines and scatters data
    prepared as in plotAll
    """

    # TODO: Background color

    colour.plotting.plot_RGB_colourspaces_in_chromaticity_diagram_CIE1931(colourspaces=['sRGB', 'DCI-P3'], standalone=False,
                        show_whitepoints = False) #, show_diagram_colours=show_diagram_colours)

    if not white_point is None:
        wp_xy = np.array(white_point)
        plt.scatter(wp_xy[0], wp_xy[1], c='k')
    
    for data,label in hue_lines:
        plt.plot(data[...,0], data[...,1], label=label, color=hue_line_color, linestyle=hue_line_style, marker=hue_point_style)
    
    cnt = 0
    for data,label in lines:
        lstyle = lineStyle(cnt)
        plt.plot(data[...,0], data[...,1], label=label, color=lstyle[0], linestyle=lstyle[2], marker=lstyle[1])
        cnt += 1
    for data,label in scatter:
        c,m = scatterStyle(cnt)
        plt.scatter(data[...,0], data[...,1], label=label, c=c, marker=m)
        cnt += 1

    plt.xlim(-0.05, 0.80)
    plt.ylim(-0.03, 0.90)

    if filename:
        filename = filename.format('xy')
    fig = colour.plotting.render(standalone=True,
                                 x_tighten=True,
                                 y_tighten=True,
                                 filename=filename,
                                 transparent_background=False)
    # if filename and not save_only:
    #     plt.show()


def plot_ycbcr709(lines = [], scatter = [], hue_lines = [], white_point = (1.0/3, 1.0/3),
                  filename = None, save_only = False):
    wp    = np.array(white_point)
    bt709 = colour.RGB_COLOURSPACES['ITU-R BT.709']
    def convertData(xyz):
        data = colour.XYZ_to_RGB(xyz, wp, bt709.whitepoint, bt709.XYZ_to_RGB_matrix, 'CAT02', cctf_encoding=colour.models.rgb.transfer_functions.eotf_inverse_BT1886)
        return colour.RGB_to_YCbCr(data, out_legal=False)

    ax  = [None] * 3
    fig = [None] * 3
    for i in range(3):
        fig[i], ax[i] = plt.subplots()
    plotHull(itur_bt709_hull[:,[1,0]], ax[0])
    plotHull(itur_bt709_hull[:,[2,0]], ax[1])
    plotHull(itur_bt709_hull[:,[1,2]], ax[2])

    ax[0].set_xlim(-0.55, 0.55)
    ax[0].set_ylim(-0.05, 1.05)
    ax[0].set_aspect('equal')
    ax[0].set_xlabel('Cb')
    ax[0].set_ylabel('Y')

    ax[1].set_xlim(-0.55, 0.55)
    ax[1].set_ylim(-0.05, 1.05)
    ax[1].set_aspect('equal')
    ax[1].set_xlabel('Cr')
    ax[1].set_ylabel('Y')

    ax[2].set_xlim(-0.55, 0.55)
    ax[2].set_ylim(-0.55, 0.55)
    ax[2].set_aspect('equal')
    ax[2].set_xlabel('Cb')
    ax[2].set_ylabel('Cr')

    for line,label in hue_lines:
        data = convertData(line)
        ax[0].plot(data[:,1], data[:,0], label=label, color=hue_line_color, linestyle=hue_line_style, marker=hue_point_style)
        ax[1].plot(data[:,2], data[:,0], label=label, color=hue_line_color, linestyle=hue_line_style, marker=hue_point_style)
        ax[2].plot(data[:,1], data[:,2], label=label, color=hue_line_color, linestyle=hue_line_style, marker=hue_point_style)

    cnt = 0
    for line,label in lines:
        data = convertData(line)
        lstyle = lineStyle(cnt)
        ax[0].plot(data[:,1], data[:,0],  label=label, color=lstyle[0], linestyle=lstyle[2], marker=lstyle[1])
        ax[1].plot(data[:,2], data[:,0],  label=label, color=lstyle[0], linestyle=lstyle[2], marker=lstyle[1])
        ax[2].plot(data[:,1], data[:,2],  label=label, color=lstyle[0], linestyle=lstyle[2], marker=lstyle[1])
        cnt += 1

    for points,label in scatter:
        data = convertData(points)
        c,m = scatterStyle(cnt)
        ax[0].scatter(data[:,1], data[:,0], label=label, c=c, marker=m)
        ax[1].scatter(data[:,2], data[:,0], label=label, c=c, marker=m)
        ax[2].scatter(data[:,1], data[:,2], label=label, c=c, marker=m)
        cnt += 1

    modes = ['Y-Cb', 'Y-Cr', 'Cr-Cb']
    for i in range(3):
        if filename:
            fig[i].savefig(filename.format('ycbcr-' + modes[i]), bbox_inches='tight')
        if not filename or not save_only:
            fig[i].show()


def plot_ictcp(lines = [], scatter = [], hue_lines = [], white_point = (1.0/3, 1.0/3),
               filename = None, save_only = False):
    wp     = np.array(white_point)
    bt2020 = colour.RGB_COLOURSPACES['ITU-R BT.2020']
    def convertData(xyz):
        data = 100 * colour.XYZ_to_RGB(xyz, wp, bt2020.whitepoint, bt2020.XYZ_to_RGB_matrix, 'CAT02', cctf_encoding=None)
        return colour.RGB_to_ICTCP(data)

    ax  = [None] * 3
    fig = [None] * 3
    for i in range(3):
        fig[i], ax[i] = plt.subplots()
    plotHull(ictcp_hull[:,[1,0]], ax[0])
    plotHull(ictcp_hull[:,[2,0]], ax[1])
    plotHull(ictcp_hull[:,[1,2]], ax[2])

    ax[0].set_xlim(-0.55, 0.55)
    ax[0].set_ylim(-0.05, 1.05)
    ax[0].set_aspect('equal')
    ax[0].set_xlabel('Ct')
    ax[0].set_ylabel('I')

    ax[1].set_xlim(-0.55, 0.55)
    ax[1].set_ylim(-0.05, 1.05)
    ax[1].set_aspect('equal')
    ax[1].set_xlabel('Cp')
    ax[1].set_ylabel('I')

    ax[2].set_xlim(-0.55, 0.55)
    ax[2].set_ylim(-0.55, 0.55)
    ax[2].set_aspect('equal')
    ax[2].set_xlabel('Ct')
    ax[2].set_ylabel('Cp')

    for line,label in hue_lines:
        data = convertData(line)
        ax[0].plot(data[:,1], data[:,0], label=label, color=hue_line_color, linestyle=hue_line_style, marker=hue_point_style)
        ax[1].plot(data[:,2], data[:,0], label=label, color=hue_line_color, linestyle=hue_line_style, marker=hue_point_style)
        ax[2].plot(data[:,1], data[:,2], label=label, color=hue_line_color, linestyle=hue_line_style, marker=hue_point_style)

    cnt = 0
    for line,label in lines:
        data = convertData(line)
        lstyle = lineStyle(cnt)
        ax[0].plot(data[:,1], data[:,0], label=label, color=lstyle[0], linestyle=lstyle[2], marker=lstyle[1])
        ax[1].plot(data[:,2], data[:,0], label=label, color=lstyle[0], linestyle=lstyle[2], marker=lstyle[1])
        ax[2].plot(data[:,1], data[:,2], label=label, color=lstyle[0], linestyle=lstyle[2], marker=lstyle[1])
        cnt += 1

    for points,label in scatter:
        data = convertData(points)
        c,m = scatterStyle(cnt)
        ax[0].scatter(data[:,1], data[:,0], label=label, c=c, marker=m)
        ax[1].scatter(data[:,2], data[:,0], label=label, c=c, marker=m)
        ax[2].scatter(data[:,1], data[:,2], label=label, c=c, marker=m)
        cnt += 1

    modes = ['I-Ct', 'I-Cp', 'Cp-Ct']
    for i in range(3):
        if filename:
            fig[i].savefig(filename.format('ictcp-' + modes[i]), bbox_inches='tight')
        if not filename or not save_only:
            fig[i].show()


def plot_jzazbz(lines = [], scatter = [], hue_lines = [], white_point = (1.0/3, 1.0/3),
                filename = None, save_only = False):
    XYZ_w  = colour.xy_to_XYZ(np.array(white_point)) * 100
    XYZ_wr = colour.xy_to_XYZ(np.array([0.3127,   0.3290])) * 100
    def convertData(xyz):
        xyz_d65 = colour.adaptation.chromatic_adaptation_VonKries(xyz * 100, XYZ_w, XYZ_wr, transform='CAT02')
        return colour.XYZ_to_JzAzBz(xyz_d65)

    ax  = [None] * 3
    fig = [None] * 3
    for i in range(3):
        fig[i], ax[i] = plt.subplots()
    plotHull(jzazbz_hull[:,[1,0]], ax[0])
    plotHull(jzazbz_hull[:,[2,0]], ax[1])
    plotHull(jzazbz_hull[:,[1,2]], ax[2])

    ax[0].set_xlim(-0.45, 0.45)
    ax[0].set_ylim(-0.05, 1.05)
    ax[0].set_aspect('equal')
    ax[0].set_xlabel('Az')
    ax[0].set_ylabel('Jz')

    ax[1].set_xlim(-0.45, 0.45)
    ax[1].set_ylim(-0.05, 1.05)
    ax[1].set_aspect('equal')
    ax[1].set_xlabel('Bz')
    ax[1].set_ylabel('Jz')

    ax[2].set_xlim(-0.45, 0.45)
    ax[2].set_ylim(-0.45, 0.45)
    ax[2].set_aspect('equal')
    ax[2].set_xlabel('Az')
    ax[2].set_ylabel('Bz')

    for line,label in hue_lines:
        data = convertData(line)
        ax[0].plot(data[:,1], data[:,0], label=label, color=hue_line_color, linestyle=hue_line_style, marker=hue_point_style)
        ax[1].plot(data[:,2], data[:,0], label=label, color=hue_line_color, linestyle=hue_line_style, marker=hue_point_style)
        ax[2].plot(data[:,1], data[:,2], label=label, color=hue_line_color, linestyle=hue_line_style, marker=hue_point_style)

    cnt = 0
    for line,label in lines:
        data = convertData(line)
        lstyle = lineStyle(cnt)
        ax[0].plot(data[:,1], data[:,0], label=label, color=lstyle[0], linestyle=lstyle[2], marker=lstyle[1])
        ax[1].plot(data[:,2], data[:,0], label=label, color=lstyle[0], linestyle=lstyle[2], marker=lstyle[1])
        ax[2].plot(data[:,1], data[:,2], label=label, color=lstyle[0], linestyle=lstyle[2], marker=lstyle[1])
        cnt += 1

    for points,label in scatter:
        data = convertData(points)
        c,m = scatterStyle(cnt)
        ax[0].scatter(data[:,1], data[:,0], label=label, c=c, marker=m)
        ax[1].scatter(data[:,2], data[:,0], label=label, c=c, marker=m)
        ax[2].scatter(data[:,1], data[:,2], label=label, c=c, marker=m)
        cnt += 1

    modes = ['Jz-Az', 'Jz-Bz', 'Bz-Az']
    for i in range(3):
        if filename:
            fig[i].savefig(filename.format('jaz-' + modes[i]), bbox_inches='tight')
        if not filename or not save_only:
            fig[i].show()


def plotAll(lines = None, scatter = None, hue_lines = None,
            white_point = (1.0/3, 1.0/3), plots = 'all',
            base_filename = None, save_only = False):
    """
    Plot different plots all at once allowing to print lines and scatter data
    and saving the plots for external use.

    linse/scatter:  List of tuples or a single tuple (nparray, label) or
                    a tuple or ndarray only.
                    XYZ data with X/Y/Z in the last dimension.
    white_point:    White point of the XYZ data
    plots:          can be a list of kinds of plots, allowed are 'uv', 'xy', 'rec709',
                    'ictcp' and 'jzazbz'. A single plot can be given as an argument.
                    'all' gives all plots
    base_filename:  Filename to save the plots. "basename_{}.png" as a replacement
                    whith the plot name is applied.
    """

    def prepInput(data):
        res_data = []
        if isinstance(data, (np.ndarray, np.generic)):
            res_data.append((data,None))
        elif isinstance(data, tuple) and isinstance(data[0], (np.ndarray, np.generic)):
            label=None
            if len(data) > 1 and isinstance(data[1], str):
                label=data[1]
            res_data.append((data[0], label))
        elif isinstance(data, list):
            for el in data:
                if isinstance(el, (np.ndarray, np.generic)):
                    res_data.append((el,None))
                elif isinstance(el, tuple) and isinstance(el[0], (np.ndarray, np.generic)):
                    label=None
                    if len(el) > 1 and isinstance(el[1], str):
                        label=el[1]
                    res_data.append((el[0], label))
        return res_data


    if plots == 'all':
        plots = ['uv', 'xy', 'rec709', 'ictcp', 'jzazbz']

    if hue_lines == 'hung':
        tmp = getHungBernsData()
        hue_lines = [(tmp[0][x], tmp[1][x])  for x in range(0, len(tmp[0]), 2)]
    elif hue_lines == 'munsell':
        hue_lines = getMunsellData()

    intern_lines     = prepInput(lines)
    intern_scatter   = prepInput(scatter)
    intern_hue_lines = prepInput(hue_lines)

    xy_lines     = []
    xy_scatter   = []
    xy_hue_lines = []
    for xyz,label in intern_lines:
        xy_lines.append((colour.XYZ_to_xy(xyz, np.array(white_point)), label))
    for xyz,label in intern_scatter:
        xy_scatter.append((colour.XYZ_to_xy(xyz, np.array(white_point)), label))
    for xyz,label in intern_hue_lines:
        xy_hue_lines.append((colour.XYZ_to_xy(xyz, np.array(white_point)), label))
    
    if 'uv' in plots:
        plot_uv(lines=xy_lines, scatter=xy_scatter, hue_lines=xy_hue_lines, white_point=white_point,
                filename=base_filename, save_only=save_only)
    if 'xy' in plots:
        plot_xy(lines=xy_lines, scatter=xy_scatter, hue_lines=xy_hue_lines, white_point=white_point,
                filename=base_filename, save_only=save_only)

    if 'rec709' in plots:
        plot_ycbcr709(lines=intern_lines, scatter=intern_scatter, hue_lines=intern_hue_lines, white_point=white_point,
                      filename=base_filename, save_only=save_only)
    if 'ictcp' in plots:
        plot_ictcp(lines=intern_lines, scatter=intern_scatter, hue_lines=intern_hue_lines, white_point=white_point,
                   filename=base_filename, save_only=save_only)
    if 'jzazbz' in plots:
        plot_jzazbz(lines=intern_lines, scatter=intern_scatter, hue_lines=intern_hue_lines, white_point=white_point,
                    filename=base_filename, save_only=save_only)