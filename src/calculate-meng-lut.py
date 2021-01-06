#!/usr/bin/python

"""
Command line tool to generate a Meng+Sat lut based on a (camera) spectral response
"""

from saturation.meng import Meng
import time
import argparse
import datetime
import sys


parser = argparse.ArgumentParser(description='Create Meng+Sat LUT from a spectral response file')
parser.add_argument('--response', '-r', dest='response', help='(Camera) spectral response')
parser.add_argument('--output-folder', '-o', dest='output_folder', help='Output folder for the LUT')
parser.add_argument('--use-uv', '-u', dest='is_uv', action='store_true', help='Create UV lut (input chromaticity goes through xy2uv conversion). Mostly intended for CIE1931 response.')
parser.add_argument('--no-compact', '-C', dest='no_compact', action='store_true', help='Don\'t create compact LUT (restricted to spectra locus based on response)')
parser.add_argument('--lut-dims', '-d', dest='dimensions', type=int, default=128, help='LUT dimension')
parser.add_argument('--max-iter', '-i', dest='max_iter', type=int, default=2500, help='Maximum number of iterations')

args = vars(parser.parse_args())
print(args)

compact = True
if args['no_compact']:
    compact = False

print('Starting process...', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

obj = Meng(args['response'], lut_dim_2d=args['dimensions'], compact_lut=compact, is_uv=args['is_uv'])
obj.maxiter = args['max_iter']
if obj.calculateLut():
    if obj.save(args['output_folder']):
        print('Success in creating the LUT')
        print('Finished at ', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        sys.exit(0)
    else:
    	print('FAILED TO SAVE')
else:
	print('FAILED TO CALCULATE')
print('')

print('         -  !!!!!  -')
print('FAILED TO CREATE LUT')
print('         -  !!!!!  -')
sys.exit(1)

