
"""
LUT calculation tool to create a spectral 2D LUT for camera to spectra characterization
"""


import math
import re
import datetime
import os
from shutil import copyfile
import subprocess
import sys
import argparse

from upsampler import minimizer
from upsampler import spectra_loader
import matplotlib.pyplot as plt
import numpy as np
from numpy import matlib


# Command line argument parsing
parser = argparse.ArgumentParser(description='Calculate 2D LUT')
parser.add_argument('processfolder', help='Folder where the LUT calculation folder should land')
parser.add_argument('cameraresponse', help='Camera response CSV')
parser.add_argument('--comment', dest='comment', default='', help='Comment/Notes for inclusion into folder name')
parser.add_argument('--plot', '-p', dest='plot', action='store_true', help='Show camera plots?')
parser.add_argument('--intermediate', '-i', dest='intermediate', action='store_true', help='Only calculate inner "triangle"')
parser.add_argument('--dimension', '-d', dest='dim', type=int, default=128, help='LUT dimension')
parser.add_argument('--maxiter', dest='maxiter', type=int, default=2000, help='Maximum iterations in minimization step')
parser.add_argument('--ftol', dest='ftol', type=float, default=1e-10, help='Precision goal for the value of f in the stopping criterion in minimization')
parser.add_argument('--threads', '-t', dest='threads', type=int, default=-1, help='Number of CPU threads to use. -1 is all available threads will be used.')
parser.set_defaults(plot=False)
parser.set_defaults(intermediate=False)

args = vars(parser.parse_args())

# Check for existens of files/paths
if not os.path.isdir(args['processfolder']):
    print('The given lut directory "' + args['processfolder'] + '" doesn\'t exist')
    sys.exit(1)
if not os.path.isfile(args['cameraresponse']):
    print('The camera response path "' + args['cameraresponse'] + '" doesn\'t exist')
    print('Exiting')
    sys.exit(1)


def git_string(mode = 'full'):
    """
    Helper to document code revision in LOG file
    """
    branch_name = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip()
    short_hash = subprocess.check_output(['git', 'log', '--format=%h', '-n', '1']).strip()
    full_hash = subprocess.check_output(['git', 'log', '--format=%H', '-n', '1']).strip()
    commit_date = subprocess.check_output(['git', 'log', '--format=%ai', '-n', '1']).strip()
    commit_author = subprocess.check_output(['git', 'log', '--format=%an', '-n', '1']).strip()
    commit_email = subprocess.check_output(['git', 'log', '--format=%ae', '-n', '1']).strip()

    if mode == 'id':
        return short_hash
    else:
        return "%s @ %s (%s - %s - %s)" % (branch_name, short_hash, commit_date, commit_author, commit_email)


# Get basic arguments
lut_dimension = args['dim']
maxiter       = args['maxiter']
ftol          = args['ftol']
spectrafile   = args['cameraresponse']

# Create folder name
tstamp = re.sub(r':', '-', str(datetime.datetime.now()))
comment = args['comment'] + '_' + str(lut_dimension) + '_' + str(maxiter) + '_' + str(ftol)
folder = 'process/' + re.sub(r'[^a-zA-Z0-9_.-]', '_', tstamp + '_' + comment)
os.makedirs(folder)
folder += '/'

# Copy files
copyfile(spectrafile, folder + os.path.basename(spectrafile))
profile = os.path.realpath(__file__)
copyfile(profile, folder + os.path.basename(profile))

# Create protocoll file
protocoll = open(folder + 'protocoll.txt', 'w')

def log(msg, flush = False):
    """
    Helper to write data to logfile and print output
    """
    protocoll.write(msg + '\n')
    print(msg)

    if flush:
        protocoll.flush()

log(str(datetime.datetime.now()) + '   Starting')
log('    git-version:   ' + git_string())
log('    lut_dimension: ' + str(lut_dimension))
log('    maxiter:       ' + str(maxiter))
log('    ftol:          ' + str(ftol), True)

# Load spectra and setup base information
arr = spectra_loader.interpolate(spectra_loader.read_spectra(spectrafile))
up = minimizer.MinimizingUpsampler()
up.setResponse(arr)

if args['threads'] > 0:
    up.threads = args['threads']

np.savetxt(folder + os.path.basename(spectrafile) + '.used.csv', up.response, delimiter=',')

# plot camera response if plot flag is set
if args['plot']:
    plt.plot(up.response[:,0], up.response[:,1], 'r')
    plt.plot(up.response[:,0], up.response[:,2], 'g')
    plt.plot(up.response[:,0], up.response[:,3], 'b')
    plt.show()

log('    start wl:      ' + str(up.response[0, 0]) + 'nm')
log('    end wl:        ' + str(up.response[-1, 0]) + 'nm')
log('')
log(str(datetime.datetime.now()) + '   Build hull and estimate number of entries', True)

# Build hull and get number of points to fil
up.buildHull()
up.maxiter = maxiter
up.ftol = ftol
number_of_points = up.estimateTriangleSize(lut_dimension)


# Plot laser spectra pq-Response if requested
if args['plot']:
    lasers = up.pqLasers()
    plt.figure()
    plt.ylim((0, 1))
    plt.xlim((0, 1))
    plt.scatter(lasers[:,0], lasers[:,1])
    plt.show()


# Inner LUT fitting
log('    Points to fit: ' + str(number_of_points))
log(str(datetime.datetime.now()) + '   Fitting ...', True)

up.fillInnerLut()

log(str(datetime.datetime.now()) + '   Inner points fittet, saving now ...', True)

np.save(folder + 'gridfile.npy', up.lut_grid)
np.save(folder + 'donefile.npy', up.lut_done)

if args['intermediate']:
    log('Called with intermediate flag, stopping here after fitting', True)
    sys.exit(0)


# postprocess (fill outside of LUT with extrapolation [nearest neighbour])
log(str(datetime.datetime.now()) + '   Fill missing LUT points ...', True)
up.fillMissing()

log(str(datetime.datetime.now()) + '   LUT build, saving and exiting ...', True)
np.save(folder + 'lut_full.npy', up.lut_grid)

log(str(datetime.datetime.now()) + '   Process done', True)
protocoll.close()
