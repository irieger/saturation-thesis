import math
import yaml
import time
import os
import pathlib
from enum import Enum
from abc import ABC, abstractmethod

import numpy as np
from numpy import matlib
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from scipy.interpolate import griddata

from multiprocessing import sharedctypes
from multiprocessing import Process, current_process, Queue

from .spectra_loader import read_cie1931, read_spectra, reduce_spectra, reduce_to_wavelengths, spectra_integrate


class FitterState(Enum):
    INIT = 0
    READY = 1
    BASE_FIT = 2
    FILLED = 3


class OptimizerBase(ABC):

    def __init__(self, lut_dim_2d = 32, compact_lut = True, is_uv = True):
        """
        Initialize base fitting structure
        """
        self.fitter_state     = FitterState.INIT
        self.lut_data         = None
        self.lut_done         = None
        self.points_to_solve  = []
        self.xyz              = read_cie1931()
        self.response         = False
        self.is_uv            = is_uv

        self.lut_min_max      = np.array([0.0, 1.0])
        self.lut_dim_2d       = lut_dim_2d
        self.lut_entry_values = 3
        self.compact_lut      = compact_lut
        self.log              = ''

        self.thread_count     = os.cpu_count()
        self.maxiter          = 2000
        self.ftol             = 1e-10
        self.spectra_epsilon  = 1e-8
        self.rgbsum           = 1

    @abstractmethod
    def save(self, folder):
        pass

    @abstractmethod
    def load(folder):
        pass

    @abstractmethod
    def _prepareOptimizer(self):
        pass

    @abstractmethod
    def fit(self, default = None, xy = None, xyz = None, uv = None):
        pass

    @abstractmethod
    def lookup(self, **kwargs):
        pass

    @abstractmethod
    def saturate(self, data, saturation):
        pass

    @abstractmethod
    def toXyz(self, data):
        pass

    def toCoordinates(self, xyz = None, xy = None, uv = None, s_ = None):
        pq = None
        s  = s_
        if not xyz is None:
            xy = OptimizerBase.xyzToXy(xyz)
            if s_ is None:
                s  = np.sum(xyz, axis=-1)
            if self.is_uv:
                pq = OptimizerBase.xyToUv(xy)
            else:
                pq = xy
        elif not xy is None:
            if self.is_uv:
                pq = OptimizerBase.xyToUv(xy)
            else:
                pq = xy
        elif not uv is None:
            if self.is_uv:
                pq = uv
            else:
                pq = OptimizerBase.uvToXy(uv)
        
        return (pq, s)


    def bilinearInterpolation(self, data):
        if self.fitter_state.value < FitterState.BASE_FIT.value:
            print('LUT not ready for lookup')
            return None
        
        x = (data[...,0] - self.lut_min_max[0]) / (self.lut_min_max[1] - self.lut_min_max[0]) * (self.lut_dim_2d - 1)
        y = (data[...,1] - self.lut_min_max[0]) / (self.lut_min_max[1] - self.lut_min_max[0]) * (self.lut_dim_2d - 1)
        
        x_low = np.int16(np.clip(x, 0, self.lut_dim_2d - 2))
        y_low = np.int16(np.clip(y, 0, self.lut_dim_2d - 2))
        x_weight = x - x_low
        y_weight = y - y_low

        res = np.zeros(data.shape[:-1] + (self.lut_entry_values,))
        foo = (1 - y_weight[...,None]) * ((1 - x_weight[...,None]) * self.lut_data[x_low, y_low,     :] + x_weight[...,None] * self.lut_data[x_low + 1, y_low,     :])
        res[...]  = (1 - y_weight[...,None]) * ((1 - x_weight[...,None]) * self.lut_data[x_low, y_low,     :] + x_weight[...,None] * self.lut_data[x_low + 1, y_low,     :])
        res[...] += y_weight[...,None]       * ((1 - x_weight[...,None]) * self.lut_data[x_low, y_low + 1, :] + x_weight[...,None] * self.lut_data[x_low + 1, y_low + 1, :])
        return res

    def prepareOptimizer(self):
        if self.fitter_state.value >= FitterState.READY.value:
            return False
        
        self.createHull()
        self.pointsToSolve()
        return self._prepareOptimizer()

    def calculateInner(self):
        if self.fitter_state.value != FitterState.READY.value:
            print('Optimizer not ready!')
            return False
        
        grid = np.ctypeslib.as_ctypes(np.zeros((self.lut_dim_2d, self.lut_dim_2d, self.lut_entry_values)))
        grid_shared = sharedctypes.RawArray(grid._type_, grid)
        done = np.ctypeslib.as_ctypes(np.full((self.lut_dim_2d, self.lut_dim_2d), False, dtype=bool))
        done_shared = sharedctypes.RawArray(done._type_, done)

        # worker process for fitting, calculates next grid point from queue until queue
        # delivers stop signal
        def worker(wnum, input_queue, result_queue):
            lgrid = np.ctypeslib.as_array(grid_shared)
            ldone = np.ctypeslib.as_array(done_shared)

            dim_mult = (self.lut_min_max[1] - self.lut_min_max[0]) / (self.lut_dim_2d - 1)
            dim_base = self.lut_min_max[0]

            os.sched_setaffinity(0, [wnum])
            while True:
                try:
                    idx, value = input_queue.get(block=False)
                    if value == 'STOP':
                        break
                    xx = dim_base + value[0] * dim_mult
                    yy = dim_base + value[1] * dim_mult
                    #print(xx, yy)
                    success = False
                    try:
                        fit_res = self.fit(default=[xx, yy])
                        if fit_res[1] == False:
                            success = True
                            lgrid[value[0], value[1], :] = np.array(fit_res[0])
                            ldone[value[0], value[1]] = True
                    except:
                        pass
                    result_queue.put((idx, success))
                except:
                    pass
                os.sched_yield()

        num_procs  = self.thread_count
        task_queue = Queue(2*num_procs)
        done_queue = Queue(2*num_procs)

        # Launch workers.
        print('Running {} workers ...'.format(num_procs))
        processes = []
        for i in range(num_procs):
            processes.append(Process(target = worker,
                args = (i, task_queue, done_queue),
                name = 'worker {}'.format(i),
                daemon = True))
            processes[-1].start()

        num_sent    = 0
        num_done    = 0
        num_clipped = 0
        num_failed  = 0
        iterator    = iter(self.points_to_solve)
        perc        = 0
        data_size   = len(self.points_to_solve)

        # Push grid points to process and ceep count. When done send stop signal
        def print_progress(msg=None):
            msg_str = ''
            if msg is not None:
                msg_str = '['+msg+']'
            print('\033[2K\r{} sent, {} done, {} clipped, {} failed, {} total ({} %) {}'.format(num_sent,
                num_done, num_clipped, num_failed, data_size, perc, msg_str), end='')

        while num_done < data_size:
            print_progress('sending work')

            while num_sent < data_size and not task_queue.full():
                nextval = next(iterator)
                task_queue.put((num_sent, nextval))
                num_sent += 1
                os.sched_yield()

            while True:
                try:
                    i, success = done_queue.get(block=False)
                    num_done += 1
                    if not success:
                        num_failed += 1
                    perc = int(num_done / data_size * 100)
                except:
                    break
                time.sleep(0)

            print_progress()
            time.sleep(2)
        
        # Terminate workers.
        for i in range(num_procs):
            task_queue.put((-1, 'STOP'))

        for p in processes:
            p.join()

        print('\n ... done')

        # Copy grid data from shared memory to instance variable
        self.lut_data = np.ctypeslib.as_array(grid_shared)
        self.lut_done = np.ctypeslib.as_array(done_shared)

        self.fitter_state = FitterState.BASE_FIT
        return True
    
    def postOptimizer(self):
        grid = np.zeros((self.lut_dim_2d, self.lut_dim_2d, 2))
        grid[...,0] = np.tile(np.linspace(self.lut_min_max[0], self.lut_min_max[1], num=self.lut_dim_2d)[:,None], (1, self.lut_dim_2d))
        grid[...,1] = grid[...,0].T
        data = self.lut_data.copy()
        done = self.lut_done.copy()
        #new_data = np.zeros(self.lut_data.shape)

        # fill inside by 
        for i in range(self.lut_entry_values):
            dtmp = data[..., i]
            new_data = griddata((grid[...,0][done].flatten(), grid[...,1][done].flatten()),
                                dtmp[done].flatten(),
                                (grid[...,0].flatten(), grid[...,1].flatten()),
                                method='nearest')
            new_data = np.reshape(new_data, dtmp.shape)
            dtmp[~done]  = new_data[~done]
            data[..., i] = dtmp
        done[~done]   = True
        self.lut_data = data.copy()
        self.lut_done = done.copy()
        self.fitter_state = FitterState.FILLED

    def _saveBase(self, folder, fill_struct = {}, overwrite = False):
        if os.path.exists(folder) and not overwrite:
            print('Folder already exists')
            return False
        
        fitter_state = self.fitter_state.value
        #if fitter_state < FitterState.BASE_FIT.value:
        #    fitter_state = FitterState.INIT
        response = None
        if not self.response is None:
            response = self.response.tolist()

        # TODO: Fill with all values from base class besides the LUT arrays!
        base_struct = {
            'fitter_state':     fitter_state,
            'points_to_solve':  self.points_to_solve,
            'xyz':              self.xyz.tolist(),
            'response':         response,
            'is_uv':            self.is_uv,
            'lut_min_max':      self.lut_min_max.tolist(),
            'lut_dim_2d':       self.lut_dim_2d,
            'lut_entry_values': self.lut_entry_values,
            'compact_lut':      self.compact_lut,
            'maxiter':          self.maxiter,
            'ftol':             self.ftol,
            'spec_eps':         self.spectra_epsilon,
            'rgbsum':           self.rgbsum
        }
        base_struct.update(fill_struct)

        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(folder, 'info.yml'), 'w') as f:
            f.write(yaml.dump(base_struct))
        with open(os.path.join(folder, 'log.txt'), 'w') as f:
            f.write(self.log)
        if not self.lut_data is None:
            np.savez_compressed(os.path.join(folder, 'lut_data.npz'), self.lut_data)
            np.savez_compressed(os.path.join(folder, 'lut_done.npz'), self.lut_done)
        return True

    def _loadBase(self, folder):
        data = None
        try:
            with open(os.path.join(folder, 'info.yml'), 'r') as f:
                data = yaml.load(f)
                self.fitter_state     = FitterState(data['fitter_state'])
                self.points_to_solve  = data['points_to_solve']
                self.xyz              = np.array(data['xyz'])
                if 'response' in data and data['response']:
                    self.response     = np.array(data['response'])
                self.is_uv            = data['is_uv']
                self.lut_min_max      = np.array(data['lut_min_max'])
                self.lut_dim_2d       = data['lut_dim_2d']
                self.lut_entry_values = data['lut_entry_values']
                self.compact_lut      = data['compact_lut']
                self.maxiter          = data['maxiter']
                self.ftol             = data['ftol']
                self.spectra_epsilon  = data['spec_eps']
                self.rgbsum           = data['rgbsum']
            with open(os.path.join(folder, 'log.txt'), 'r') as f:
                self.log = f.read()
            if os.path.exists(os.path.join(folder, 'lut_data.npz')) and os.path.exists(os.path.join(folder, 'lut_done.npz')):
                self.lut_data = np.load(os.path.join(folder, 'lut_data.npz'))['arr_0']
                self.lut_done = np.load(os.path.join(folder, 'lut_done.npz'))['arr_0']
        except:
            return False
        return data

    def calculateLut(self, auto_prep = True, auto_post = True):
        if self.fitter_state.value < FitterState.READY.value and auto_prep:
            self.prepareOptimizer()
        
        if not self.calculateInner():
            print('Fitter not ready or an error occured')
            return False
        
        if auto_post:
            return self.postOptimizer()
        
        return True

    def createHull(self):
        resp = self.xyz
        if not self.response is None:
            resp = self.response
        monochromatics = np.identity(resp.shape[0])
        monochromatics = spectra_integrate(monochromatics, resp[:,1:])
        pts = OptimizerBase.xyzToXy(monochromatics)
        if self.is_uv:
            pts = OptimizerBase.xyToUv(pts)
        convex_hull = ConvexHull(pts)
        self.delaune_points = pts[convex_hull.vertices, :]
        self.delaune = Delaunay(self.delaune_points)

    def pointsToSolve(self):
        if self.compact_lut:
            self.lut_min_max[0] = np.min(self.delaune_points)
            self.lut_min_max[1] = np.max(self.delaune_points)
            dist = (self.lut_min_max[1] - self.lut_min_max[0]) / (self.lut_dim_2d - 3)
            self.lut_min_max[0] -= dist
            self.lut_min_max[1] += dist
        
        vals = np.linspace(self.lut_min_max[0], self.lut_min_max[1], num=self.lut_dim_2d)
        self.points_to_solve = []
        cnt = 0

        for x in range(self.lut_dim_2d):
            xx = vals[x]
            for y in range(self.lut_dim_2d):
                yy = vals[y]
                coord = np.array([xx, yy])
                if self.delaune.find_simplex([xx, yy]) >= 0:
                    self.points_to_solve.append([x, y])
        #print(self.points_to_solve)
        return len(self.points_to_solve)

    def processImage(self, saturation, xyz):
        if len(xyz.shape) == 2:
            data = self.lookup(xyz=xyz)
            data = self.saturate(data, saturation)
            return self.toXyz(data)
        elif len(xyz.shape) == 3:
            # As array can get quite big especially with 1nm stepped spectral upsampling
            # of larger images processing is split line wise
            res = np.zeros(xyz.shape)
            for y in range(xyz.shape[0]):
                data = self.lookup(xyz=xyz[y,...])
                data = self.saturate(data, saturation)
                res[y, ...] = self.toXyz(data)
            return res
        else:
            print('Failed to process. xyz has to be 2D or 3D matrix with nxmx3')
            return None


    def xyzToXy(data):
        s = np.sum(data, axis=1)
        s[s < 0.0000000001] = 1
        r = np.zeros((data.shape[0], 2))
        r[:, 0] = np.divide(data[:, 0], s)
        r[:, 1] = np.divide(data[:, 1], s)
        return r
    
    def xyToUv(data):
        x = data[:,0]
        y = data[:,1]
        t = -2 * x + 12 * y + 3
        
        res = np.zeros(data.shape)
        res[:,0] = 4 * x / t
        res[:,1] = 9 * y / t
        return res

    def uvToXy(data):
        u = data[:,0]
        v = data[:,1]
        t = 6 * u - 16 * v + 12

        res = np.zeros(data.shape)
        res[:,0] = 9 * u / t
        res[:,1] = 4 * v / t
        return res
        