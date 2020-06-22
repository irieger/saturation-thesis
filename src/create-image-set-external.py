#!/usr/bin/python

"""
Modified hack to use an external script to process the images in a C++/CUDA tool
for faster image processing as the current implementation of KSM and Meng+Sat have been
slow and working unreliable when executed in the threaded environment.
"""


import os
import sys
import time
import argparse
from shutil import copyfile,rmtree
import json
import subprocess
import pathlib
import hashlib
import tempfile

# from multiprocessing import sharedctypes
from multiprocessing import Process, current_process, Queue

import numpy as np
import colour

import colorspaces
import saturation
import plotting


parser = argparse.ArgumentParser(description='Process images and create a dataset for the html saturation survey viewer')
parser.add_argument('--image-filelist', '-I', dest='image_filelist', help='Input image file list (json) - see image-set.json as an example')
parser.add_argument('--output-folder', '-O', dest='output_folder', help='Output folder for the full survey')
parser.add_argument('--plott-colors', '-P', dest='plot_colors', help='File with the primaries to calculate')
parser.add_argument('--saturation-models', '-S', dest='sat_model', help='Saturation models to calculate')
# parser.add_argument('--saturation-models', '-S', dest='sat_model', help='Saturation models to calculate')
parser.add_argument('--force', '-f', dest='force_overwrite', action='store_true', help='Force overwrite')
parser.add_argument('--no-ref', '-r', dest='no_ref', action='store_true', help='Don\'t create reference images')
# parser.add_argument('--no-data-js', '-j', dest='no_data_js', action='store_true', help='Don\'t create data.js overview file')
parser.add_argument('--batch-file', '-b', dest='create_batch_file', action='store_true', help='Create batch file')
parser.add_argument('--single-thread', '-s', dest='single_thread', action='store_true', help='Single threaded execution')
parser.add_argument('--threads', '-T', dest='threads', type=int, default=-1, help='Number of threads (-1 == auto)')
parser.add_argument('--ignore-skeleton', dest='ignore_skeleton', action='store_true', help='Don\'t create skeleton')
parser.set_defaults(no_primaries=False)

args = vars(parser.parse_args())

if not args['output_folder'] or (os.path.exists(args['output_folder']) and not args['force_overwrite']):
    print('Invalid output folder', args['output_folder'])
    print('Folder exists or not given')
    sys.exit(1)

output_folder  = args['output_folder']


saturation_steps    = np.linspace(0.0, 2.0, num=9)
# saturation_steps    = np.array([0.25])
max_num_scatter = 200000
# per image every np.ceil(np.sqrt(pixel / max_num_scatter)) th pixel per line/row

transfer_curves = [ ('sRGB - pure OETF', 'srgb_oetf'),
                    ('sRGB - tonemapped', 'srgb_hermite'),
                    ('sRGB - tonemapped (RGB)', 'srgb_hermite_rgb'),
                    ('Display P3 - pure OETF', 'disp3_oetf'),
                    ('Display P3 - tonemapped', 'disp3_hermite'),
                    ('Display P3 - tonemapped (RGB)', 'disp3_hermite_rgb')
                  ]
sat_models = [ ('YCbCr ITU-R BT.709', 'bt709'),
               ('YCbCr ITU-R BT.2020', 'bt2020'),
               ('YCbCr ITU-R BT.2100 (const luminance / ICtCp)', 'bt2100const'),
               ('JzAzBz', 'jzazbz'),
               ('ASC-CDL (Rec.709, no clamp for values > 1.0)', 'asccdl')
             ]
sat_models = [ ('KSM (Luma preserve 0.00)', 'ksm0.00'),
#                 ('KSM (Luma preserve 0.25)', 'ksm0.25'),
                ('KSM (Luma preserve 0.50)', 'ksm0.50'),
#                 ('KSM (Luma preserve 0.75)', 'ksm0.75'),
                ('KSM (Luma preserve 1.00)', 'ksm1.00') ]
sat_models += [ ('Meng+Sat (Luma preserve 0.00)', 'meng0.00'),
                ('Meng+Sat (Luma preserve 0.50)', 'meng0.50'),
                ('Meng+Sat (Luma preserve 1.00)', 'meng1.00') ]

if args['sat_model']:
    sat_models = []
    if os.path.exists(args['sat_model']):
        fp = open(args['sat_model'], 'r')
        sat_models = json.loads(fp.read())
        fp.close()
    else:
        sat_models = json.loads(args['sat_model'])


images = []
if args['image_filelist'] and os.path.exists(args['image_filelist']):
    fp = open(args['image_filelist'], 'r')
    images = json.loads(fp.read())
    fp.close()
elif args['image_filelist']:
    images = json.loads(args['image_filelist'])


graphs = []
if args['plot_colors'] and os.path.exists(args['plot_colors']):
    fp = open(args['plot_colors'], 'r')
    graphs = json.loads(fp.read())
    fp.close()
elif args['plot_colors']:
    graphs = json.loads(args['plot_colors'])


basepath = os.path.dirname(os.path.realpath(__file__))
basepath = os.path.abspath(os.path.join(basepath, '..'))

p3_icc_path = os.path.join(basepath, 'external/compact-icc-profiles/profiles/DisplayP3Compat-v2-magic.icc')

output_json = {}

output_json['color_spaces'] = []
for tc in transfer_curves:
    output_json['color_spaces'].append({'text': tc[0], 'mode': 'images', 'path': tc[1]})
output_json['color_spaces'].append({'text': 'Graphs', 'mode': 'graphs', 'path': 'graphs'})

output_json['saturation'] = list(saturation_steps)

output_json['color_models'] = []
for sm in sat_models:
    output_json['color_models'].append({'text': sm[0], 'path': sm[1]})

output_json['image_sets'] = {}
output_json['image_sets']['default'] = []
output_json['image_sets']['graphs'] = []


if args['create_batch_file']:
    print('Batch file -------')
    for img in images:
        ref = False
        for sat in sat_models:
            ref_str = '--no-ref'
            if not ref:
                ref_str = ''
                ref = True
            print('python src/create-image-set.py -s --ignore-skeleton -f -O \'' + args['output_folder']
                + '\' -I \'', json.dumps([img]), '\' -S \'', json.dumps([sat]), '\'', ref_str)
    for graph in graphs:
        print('python src/create-image-set.py -s --ignore-skeleton -f -O \'' + args['output_folder']
                + '\' -P \'', json.dumps([graph]), '\' -S \'', json.dumps(sat_models), '\'')
    print('------')
    sys.exit(0)


if args['force_overwrite'] and os.path.exists(args['output_folder']) and not args['ignore_skeleton']:
    print('Delete existing folder ...')
    rmtree(args['output_folder'])
if not args['ignore_skeleton']:
    copyfiles = ['css/base.css', 'js/vue.js', 'js/vue.min.js', 'index.html']
    pathlib.Path(os.path.join(output_folder, 'js')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(output_folder, 'css')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(output_folder, 'data')).mkdir(parents=True, exist_ok=True)
    for fname in copyfiles:
        copyfile(os.path.join(basepath, 'html', fname), os.path.join(output_folder, fname))



def tonemap(img_data, base_path):
    for tc in transfer_curves:
        opath   = base_path.replace('$oetf$', tc[1])

        folder = os.path.dirname(opath)
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

        data = colorspaces.outputColor(img_data, tc[1])

        write_path = opath
        p3_profile = False
        depth='uint8'
        if 'disp3' in tc[1].lower():
            p3_profile = True
            write_path = tempfile.mktemp() + '.png'
            depth='uint16'

        colour.write_image(data, write_path, bit_depth=depth, method='OpenImageIO')

        if p3_profile:
            # only add profile, no conversion. Therefore in and out profile are the same
            cmd = ['convert', write_path, '-profile', p3_icc_path, '-profile', p3_icc_path, opath]
            imagick = subprocess.run(cmd)
            if imagick.returncode != 0:
                print('Failed to convert ' + opath + '  ', params)
            os.remove(write_path)

def saturate(img_data, base_path, proc_sat_models = sat_models):
    # for model in proc_sat_models:
    #     model_path = base_path.replace('$model$', model[1])
    #     for sat in saturation_steps:
    #         opath = model_path.replace('$sat$', '%.3f' % (sat))
    #         data  = saturation.saturate(img_data, sat, model[1])
    #         tonemap(data, opath)
    #         data = None
    modes = []
    vals  = []
    for sat in sat_models:
        modes.append(sat[1])
    for sat in saturation_steps:
        vals.append(str(sat))

    tpath   = tempfile.mktemp() + '.npy'
    respath = tempfile.mktemp()
    np.save(tpath, img_data)

    cmd = [basepath + '/process_ict.bash', tpath, ':'.join(modes), ':'.join(vals), respath]
    ext_proc = subprocess.run(cmd, cwd=basepath)
    os.remove(tpath)
    if ext_proc.returncode != 0:
        print('Failed to externally process...')
        print('')
        foo = input('Kill process or press enter to continue...\n')

    # foo = input('Halting for user input ...\n')

    for model in modes:
        model_path = base_path.replace('$model$', model)
        for sat in vals:
            img_path = respath + '_' + model + '_' + sat + '.npy'
            processed_img = np.load(img_path)
            os.remove(img_path)
            sat_f = float(sat)
            opath = model_path.replace('$sat$', '%.3f' % (sat_f))
            tonemap(processed_img, opath)

def processImages(proc_images = images, proc_sat_models = sat_models, thread_mode=False):
    output_filename = os.path.join(output_folder, 'data/img_$image_hash$/$model$/$oetf$/img_$sat$.png')
    for img in proc_images:
        print('Working on image', img[0], '...')

        md5hash = hashlib.md5(img[0].encode('utf-8')).hexdigest()
        base_path = output_filename.replace('$image_hash$', md5hash)

        if not thread_mode:
            output_json['image_sets']['default'].append({ 'text': img[1],
                                                          'path': 'img_' + md5hash })

        color_space = ''
        exposure = 0
        if len(img) > 2:
            color_space = img[2]
        if len(img) > 3:
            exposure = img[3]
        data = colorspaces.importImage(img[0], color_space)
        if data is None:
            print('Image ', img[0], ' invalid!')
            continue
        data = data * np.power(2.0, exposure)

        if len(proc_sat_models) > 0:
            saturate(data, base_path, proc_sat_models)

        if len(proc_sat_models) == 0 or len(proc_sat_models) == sat_models:
            if not args['no_ref']:
                ref_path = base_path.replace('$model$', 'ref').replace('$sat$', 'ref')
                tonemap(data, ref_path)

            if thread_mode:
                return ('default', { 'text': img[1], 'path': 'img_' + md5hash })
    return (None, )


def createGraphs(data_set, base_path):
    for model in sat_models:
        model_path = base_path.replace('$model$', model[1])
        pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)

        lines = []
        for i in range(data_set.shape[0]):
            saturated = np.zeros((saturation_steps.shape[0],) + data_set.shape[1:])
            for j in range(saturation_steps.shape[0]):
                saturated[j, ...] = saturation.saturate(data_set[i,...], saturation_steps[j], model[1])
            lines.append(saturated)

        bpath = os.path.join(model_path, 'graph_{}.png')
        plotting.plotAll(lines=lines, hue_lines='hung', base_filename=bpath, save_only=True)

def processGraphs(proc_graphs = graphs, thread_mode=False):
    output_filepath = os.path.join(output_folder, 'data/graph_$graph_path$/$model$/')
    
    for graph in proc_graphs:
        print('Working on graph', graph[0], '...')

        md5hash = hashlib.md5(graph[0].encode('utf-8')).hexdigest()
        if not thread_mode:
            output_json['image_sets']['graphs'].append({ 'text': graph[0],
                                                         'path': 'graph_' + md5hash })
        opath = output_filepath.replace('$graph_path$', md5hash)

        data = graph[1]
        if isinstance(data, str):
            # check if file exists and open as image
            nth_pixel = max_num_scatter
            color_space = ''
            if len(graph) > 2:
                color_space = graph[2]
            if len(graph) > 3:
                nth_pixel = graph[3]
            img = colorspaces.importImage(graph[1], color_space)
            if img is None:
                print('Image ', graph[1], ' invalid!')
                continue
            else:
                data = img.reshape(img.shape[0] * img.shape[1], img.shape[2])[::nth_pixel, :]
        else:
            data = np.array(data)
            if len(graph) > 2:
                data = colorspaces.convertColor(data, graph[2])

        createGraphs(data, opath)
        if thread_mode:
            return ('graphs', { 'text': graph[0], 'path': 'graph_' + md5hash })

    return (None, )

if args['single_thread']:
    if images:
        processImages()
    if graphs:
        processGraphs()
else:
    def worker(wnum, input_queue, result_queue):
        os.sched_setaffinity(0, [wnum])
        # print('Start worker:', wnum)
        while True:
            # try:
            if True:
                value = input_queue.get(block=True)
                # print('got data', value)
                if value == 'STOP':
                    # print('Stopping worker', wnum)
                    break
                res = (None, )
                if value[0] == 'image':
                    timages = [value[1]]
                    tsat_models = [value[2]]
                    if value[2] is None:
                        tsat_models = []
                    # print('Working on image', timages, tsat_models)
                    res = processImages(proc_images=timages, proc_sat_models=tsat_models, thread_mode=True)
                elif value[0] == 'graph':
                    res = processGraphs([value[1]], thread_mode=True)
                else:
                    print('Failed: ', value, '\n')
                # print('Worker done with current task', wnum)
                result_queue.put(res)

            # except:
            #     pass
            os.sched_yield()

    num_procs = args['threads']
    if num_procs <= 0:
        num_procs = os.cpu_count() - 3
    task_queue = Queue(3*num_procs)
    done_queue = Queue(3*num_procs)

    print('Running {} workers ...'.format(num_procs))
    processes = []
    for i in range(num_procs):
        processes.append(Process(target = worker,
            args = (i, task_queue, done_queue),
            name = 'worker {}'.format(i),
            daemon = True))
        processes[-1].start()

    todo = []
    for img in images:
        for model in sat_models:
            todo.append(('image', img, model))
        todo.append(('image', img, None))
    for graph in graphs:
        todo.append(('graph', graph))

    # print(todo)
    # for item in todo:
    #     print(item)
    # sys.exit(1)

    num_sent = 0
    num_done = 0
    num_todo = len(todo)
    perc = 0
    iterator = iter(todo)

    # Push grid points to process and ceep count. When done send stop signal
    def print_progress(msg=None):
        msg_str = ''
        if msg is not None:
            msg_str = '['+msg+']'
        print('\033[2K\r{} sent, {} done, {} total ({} %) {}'.format(num_sent,
            num_done, num_todo, perc, msg_str), end='')

    while num_done < num_todo:
        print_progress('sending work')

        while num_sent < num_todo and not task_queue.full():
            nextval = next(iterator)
            task_queue.put(nextval)
            num_sent += 1
            os.sched_yield()

        while True:
            try:
                item = done_queue.get(block=False)
                if len(item) > 0 and (not item[0] is None):
                    output_json['image_sets'][item[0]].append(item[1])
                num_done += 1
                perc = int(num_done / num_todo * 100)
            except:
                break
            time.sleep(0)

        print_progress()
        time.sleep(10)

    # Terminate workers.
    for i in range(num_procs):
        task_queue.put('STOP')

    for p in processes:
        p.join()

    print('\n ... done')


if not args['ignore_skeleton']:
    with open(os.path.join(output_folder, 'data/dataset.js'), 'w') as outfile:
        outfile.write('survey_data_set = ')
        outfile.write(json.dumps(output_json, indent=2))