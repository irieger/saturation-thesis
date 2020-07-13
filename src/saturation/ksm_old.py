import math
import time
import os
import sys
import yaml

# Multiprocessing for shared data types as well as processing fitter in parallel to use
# one minimizer per core
from multiprocessing import sharedctypes
from multiprocessing import Process, current_process, Queue

# Numpy and scipy for numbre crunching
import numpy as np
from numpy import matlib
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from scipy.interpolate import griddata


# Colour is used for plotting of spectra and calculation of standard illuminants
import colour
from colour.plotting import *
import colour.plotting
import colour.colorimetry as colorimetry

# Our helper for loading spectra/interpolating
from .spectra_loader import read_spectra, interpolate, read_cie1931


class KSM:
    """
    Wrapper for KSM
    """

    def __init__(self, lut_dim = -1, load_folder = None, uv_grid = False, auto_fill = False):
        """
        Initialize base parameters
        """
        self.maxiter = 10000
        self.ftol    = 1e-8
        self.lut_dim = 512

        self.sigma_cnstr_min =    0.000001
        self.sigma_cnstr_max = 1500.0000000

        self.max_dist = 0.07

        self.uv_mode = uv_grid

        #self.maxiter = 500
        #self.ftol    = 1e-6
        #self.lut_dim = 64

        if lut_dim > 0:
            self.lut_dim = lut_dim

        self.threads = os.cpu_count()

        xyz          = read_cie1931()
        reduced      = xyz[20:-50, :]
        self.wlrange = reduced[:, 0]
        self.xyz     = reduced[:, 1:].T

        self.points_to_solve = []

        self.wlin       = self.wlrange.copy()
        self.wlin_u     = self.wlin[-1]
        self.wlin_l     = self.wlin[0]
        self.wlin_half  = (self.wlin_u - self.wlin_l) // 2

        self.inrange0   = self.wlin - self.wlin_l
        self.inrange0_u = self.inrange0[-1]
        self.inrange1   = self.wlin_u - self.wlin_l - self.inrange0

        self.lut_grid = None
        self.lut_done = None
        self.full_sample_set = None

        self.fit_grid = None
        self.fit_done = None
        self.indexes  = None

        if load_folder:
            if os.path.isfile(os.path.join(load_folder, 'lut_grid.npy')):
                self.lut_grid = np.load(os.path.join(load_folder, 'lut_grid.npy'))
            if os.path.isfile(os.path.join(load_folder, 'lut_done.npy')):
                self.lut_done = np.load(os.path.join(load_folder, 'lut_done.npy'))
            if os.path.isfile(os.path.join(load_folder, 'fit_grid.npy')):
                self.fit_grid = np.load(os.path.join(load_folder, 'fit_grid.npy'))
            if os.path.isfile(os.path.join(load_folder, 'fit_done.npy')):
                self.fit_done = np.load(os.path.join(load_folder, 'fit_done.npy'))
            if os.path.isfile(os.path.join(load_folder, 'full_sample_set.npy')):
                self.full_sample_set = np.load(os.path.join(load_folder, 'full_sample_set.npy'))
                i = self.full_sample_set.shape[0]
                self.indexes         = np.linspace(0, i-1, num=i, dtype=np.uint32)
            if auto_fill and np.sum(self.fit_done) != self.fit_done.shape[0]**2:
                self.fillMissing()
        else:
            print('building scatter, may take some time')
            self.buildScatter()
        self.buildHull()

    def save(self, folder):
        if os.path.isdir(folder):
            raise(Exception('Folder already exists'))
        os.mkdir(folder)
        if not os.path.isdir(folder):
            raise(Exception('Failed to create folder'))

        if not self.lut_grid is None:
            np.save(os.path.join(folder, 'lut_grid'), self.lut_grid)
        if not self.lut_done is None:
            np.save(os.path.join(folder, 'lut_done'), self.lut_done)
        if not self.fit_grid is None:
            np.save(os.path.join(folder, 'fit_grid'), self.fit_grid)
        if not self.fit_done is None:
            np.save(os.path.join(folder, 'fit_done'), self.fit_done)
        if not self.full_sample_set is None:
            np.save(os.path.join(folder, 'full_sample_set'), self.full_sample_set)


    def buildScatter(self):
        # steps_wl = 100
        start_wl = self.wlin[0]
        end_wl   = self.wlin[-1]
        size = int(end_wl-start_wl + 1.0001)
        inrange_up = np.linspace(0, end_wl - start_wl, num=size)
        inrange_down = np.absolute(inrange_up - (end_wl-start_wl))
        inrange_wl = np.linspace(start_wl, end_wl, num=size)
        steps_wl = lambda x: size + (size - 1) * (x-1)
        steps_sig = 80

        peak_space = np.linspace(start_wl, end_wl, num=steps_wl(3))
        sig_space = np.linspace(0.01, 30, num=steps_sig)
        k_space = [1, 1.35, 1.7, 2]

        # 5th dimension is x(p), y(q), peak, max, sig
        gauss_funcs = np.zeros((peak_space.shape[0] * sig_space.shape[0] * len(k_space), 4))
        maximum = 1.0

        i = 0
        cnt = 0
        sig_max = 0
        for peak in peak_space:
            for sig_val in sig_space:
                for k in k_space:
                    sig = sig_val**k  #**2
                    sig_max = max(sig_max, sig)
                    val = self.gauss([peak, sig])
                    xy = self.evalSpec(val)
                    if not np.isnan(xy[0]) and not np.isnan(xy[1]):
                        gauss_funcs[i, 0:2] = xy
                        gauss_funcs[i, 2]   = peak
                        gauss_funcs[i, 3]   = sig
                        i += 1
                    else:
                        cnt += 1
        print("invalid entries: ", cnt)

        self.full_sample_set = gauss_funcs[0:i, :].copy()
        if self.uv_mode:
            self.full_sample_set[..., :2] = KSM.xyToUv(gauss_funcs[0:i, :2])
        self.indexes         = np.linspace(0, i-1, num=i, dtype=np.uint32)


    def buildHull(self):
        pq = np.zeros((self.wlin.shape[0], 2))
        for i in range(pq.shape[0]):
            spec = np.zeros((pq.shape[0]))
            spec[i]  = 1000
            pq[i, :] = self.evalSpec(spec)
            if self.uv_mode:
                pq[i, :] = KSM.xyToUv(pq[i, :])

        self.convex_hull = ConvexHull(pq)
        self.delaune = Delaunay(pq[self.convex_hull.vertices, :])


    def inHull(self, p):
        """
        Test if points in `p` are in `hull`

        `p` should be a `NxK` coordinates of `N` points in `K` dimensions
        `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
        coordinates of `M` points in `K`dimensions for which Delaunay triangulation
        will be computed
        """
        return self.delaune.find_simplex(p) >= 0


    def plotScatter(self):
        plt.figure(figsize=(20,20))
        plt.scatter(self.full_sample_set[:,0], self.full_sample_set[:,1])
        axes = plt.gca()
        axes.set_ylim([0,1])
        axes.set_xlim([0,1])
        plt.grid()


    def getX0(self, xy):
        max_dist = self.max_dist
        mdist = 0.0
        dist  = np.ones((10))
        while np.sum(dist < mdist) < 1 and mdist < max_dist:
            mdist += 0.005
            dist   = np.sqrt(np.power(self.full_sample_set[:,0] - xy[0], 2) + np.power(self.full_sample_set[:,1] - xy[1], 2))
        if mdist >= max_dist:
            return np.array([555, 400])

        idx  = self.indexes[dist < mdist]
        midx = np.argmin(self.full_sample_set[idx, 3])
        return self.full_sample_set[idx[midx], 2:]


    def fit(self, xy):
        # Warum wieso weshalb???
        def objective(s):
            return 1 / s[1]

        x0    = self.getX0(xy)
        bnds  = [(self.wlin_l, self.wlin_u), (self.sigma_cnstr_min, self.sigma_cnstr_max)]
        cnstr = {
            'type': 'eq',
            'fun': lambda s: self.constraintFunc(s, xy)
        }
        if self.uv_mode:
            cnstr['fun'] = lambda s: self.constraintFuncUv(s, xy)
        res     = minimize(objective, x0, method='SLSQP', constraints=cnstr, bounds=bnds,
                           options={"maxiter": self.maxiter, "ftol": self.ftol})
        if not res.success:
            #print('Error for xyz={} after {} iterations: {}'.format(xy, res.nit, res.message))
            #print(res)
            return (res.x, False)
        return (res.x, True)


    def gauss(self, s):
        peak = s[0] - self.wlin_l
        vrange = np.absolute(peak - self.inrange0)
        wrange = None
        if peak < self.wlin_half:
            wrange = self.inrange1 + peak
        else:
            wrange = np.absolute(self.inrange0 + (self.inrange0_u - peak))

        mrange = np.minimum(vrange, wrange)
        gvals  = np.exp(-np.power(mrange, 2.) / (np.power(s[1], 2.)))
        return gvals

    def gaussNp(self, vals):
        peak = vals[...,0] - self.wlin_l
        peak = np.repeat(peak[..., None], self.inrange1.shape[-1], axis=-1)
        stddev = vals[...,1]
        stddev[stddev < 0.000001] = 0.000001
        stddev = np.repeat(vals[...,1, None], self.inrange1.shape[-1], axis=-1)
        vrange = np.absolute(peak - self.inrange0)
        idx = peak < self.wlin_half
        wrange = np.zeros(vrange.shape)
        if np.sum(idx) > 0:
            wrange[idx]  = peak[idx] + np.tile(self.inrange1, (idx[idx].shape[0] // self.inrange1.shape[0],))
        if np.sum(~idx) > 0:
            wrange[~idx] = np.absolute(-peak[~idx] + np.tile(self.inrange0 + self.inrange0_u, (idx[~idx].shape[0] // self.inrange1.shape[0],)))
        mrange = np.minimum(vrange, wrange)
        return np.exp(-np.power(mrange, 2.) / (np.power(stddev, 2.)))


    def evalSpec(self, gvals):
        xyz = np.sum(np.multiply(gvals, self.xyz), axis=1) * 10
        s = np.sum(xyz)
        return np.array([xyz[0] / s, xyz[1] / s]).T

    def evalSpecXyz(self, gvals=None, gparams=None):
        if not gparams is None:
            # Compromise part as RAM usage is very high when a full width x height image with 401 spectral values
            # is calculated
            res = np.zeros(gparams.shape[:-1] + (3,))
            for i in range(gparams.shape[0]):
                res[i, ...] = self.evalSpecXyz(gvals=self.gaussNp(gparams[i,...]))
            return res

        res = np.zeros(gvals.shape[:-1] + (3,))
        for i in range(3):
            res[...,i] = np.sum(np.multiply(gvals, self.xyz[i,:]), axis=-1) * 10
        return res

    
    def xyToUv(xy_values):
        x = xy_values[...,0]
        y = xy_values[...,1]
        t = -2 * x + 12 * y + 3
        
        res = np.zeros(xy_values.shape)
        res[...,0] = 4 * x / t
        res[...,1] = 9 * y / t
        return res
    

    def uvToXy(uv_values):
        u = uv_values[...,0]
        v = uv_values[...,1] / 1.5
        t = 2 * u - 8 * v + 4

        res = np.zeros(uv_values.shape)
        res[...,0] = 3 * u / t
        res[...,1] = 2 * v / t
        return res


    def evalUv(self, gvals):
        return KSM.xyToUv(self.evalSpec(gvals))


    def constraintFunc(self, s, target):
        return self.evalSpec(self.gauss(s)) - target

    def constraintFuncUv(self, s, target):
        return KSM.xyToUv(self.evalSpec(self.gauss(s))) - target


    def estimateTriangleSize(self, lut_dimension = -1):
        """
        Estimate number of points to solve for the current convex hull for given
        LUT dimension.

        Multi threading inspired by the lut creation code in the Meng 2015 paper.
        """
        if lut_dimension > 0:
            self.lut_dim = lut_dimension
        else:
            lut_dimension = self.lut_dim
        self.points_to_solve = []
        self.lut_dim_mult = 1.0 / (lut_dimension - 1)

        cnt = 0
        for x in range(0, lut_dimension):
            for y in range(0, lut_dimension):
                xx = x * self.lut_dim_mult
                yy = y * self.lut_dim_mult
                if self.inHull([xx, yy]):
                    self.points_to_solve.append([x, y])
                    cnt += 1
        return cnt


    def fillInnerLut(self, lut_dim = -1):
        """
        Fill all point inside the LUT that result in a valid minimization.

        This step takes long depending on the LUT dimensions

        model parameter is either KSM for KSM kernel fitting (or currently meng style fit
        for every other value)
        """

        # clear LUT and Grid first
        self.lut_grid = None
        self.lut_done = None

        #num_bins = self.setupFitterModel(model)

        grid = np.ctypeslib.as_ctypes(np.zeros((self.lut_dim, self.lut_dim, 2)))
        grid_shared = sharedctypes.RawArray(grid._type_, grid)
        done = np.ctypeslib.as_ctypes(np.full((self.lut_dim, self.lut_dim), False, dtype=bool))
        done_shared = sharedctypes.RawArray(done._type_, done)

        num_procs = self.threads-1

        self.estimateTriangleSize(lut_dim)

        # worker process for fitting, calculates next grid point from queue until queue
        # delivers stop signal
        def worker(wnum, input_queue, result_queue):
            lgrid = np.ctypeslib.as_array(grid_shared)
            ldone = np.ctypeslib.as_array(done_shared)

            os.sched_setaffinity(0, [wnum])
            while True:
                try:
                    idx, value = input_queue.get(block=False)
                    if value == 'STOP':
                        break
                    xx = value[0] * self.lut_dim_mult
                    yy = value[1] * self.lut_dim_mult
                    success = False
                    try:
                        fit_res = self.fit([xx, yy])
                        if fit_res[1] == True:
                            success = True
                            lgrid[value[0], value[1], :] = np.array(fit_res[0])
                            ldone[value[0], value[1]] = True
                    except:
                        pass
                    result_queue.put((idx, success))
                except:
                    pass
                os.sched_yield()

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

        num_sent = 0
        num_done = 0
        num_clipped = 0
        num_failed = 0
        iterator = iter(self.points_to_solve)
        perc = 0

        data_size = len(self.points_to_solve)

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
                    break;
                time.sleep(0)

            print_progress()
            time.sleep(0)

        # Terminate workers.
        for i in range(num_procs):
            task_queue.put((-1, 'STOP'))

        for p in processes:
            p.join()

        print('\n ... done')

        # Copy grid data from shared memory to instance variable
        self.lut_grid = np.ctypeslib.as_array(grid_shared)
        self.lut_done = np.ctypeslib.as_array(done_shared)

        self.fit_grid = self.lut_grid.copy()
        self.fit_done = self.lut_done.copy()
    

    def peakToArc(self, peak_wl):
        """
        Convert peak wavelength value to circle arc position as xy coordinates
        """
        multiplier = 2 * np.pi / (self.wlin_u - self.wlin_l)
        alpha = (peak_wl - self.wlin_l) * multiplier
        return (np.cos(alpha), np.sin(alpha))

    def arcToPeak(self, coords):
        """
        Convert peak wavelength value to circle arc position as xy coordinates
        """
        multiplier = 2 * np.pi / (self.wlin_u - self.wlin_l)
        # x = coords.copy()
        # s = np.sqrt(np.sum(np.power(x, 2), axis=-1))
        # x[..., 0] = x[..., 0] / s
        # x[..., 1] = x[..., 1] / s
        alpha = np.array(np.arctan2(coords[..., 1], coords[..., 0]))
        alpha[alpha < 0] = 2 * np.pi + alpha[alpha < 0]
        while (np.any(alpha > 2 * np.pi)):
            alpha[alpha > 2 * np.pi] = alpha[alpha > 2 * np.pi] - 2*np.pi
        return (alpha / multiplier) + self.wlin_l


    def fillMissing(self):
        """
        Interpolate missing LUT elements based on linear interpolation (in Hull)
        and extrapolation for out of Hull values
        """
        grid = np.zeros((self.lut_dim, self.lut_dim, 2))
        grid[...,0] = np.tile(np.linspace(0, 1, num=self.lut_dim)[:,None], (1, self.lut_dim))
        grid[...,1] = grid[...,0].T

        xc,yc  = self.peakToArc(self.fit_grid[...,0])
        stddev = self.fit_grid[...,1]
        done   = self.fit_done.copy()

        linear_stddev  = griddata((grid[...,0][done].flatten(), grid[...,1][done].flatten()),
                                  stddev[done].flatten(),
                                  (grid[...,0].flatten(), grid[...,1].flatten()),
                                  method='cubic', fill_value=np.nan)
        linear_xc      = griddata((grid[...,0][done].flatten(), grid[...,1][done].flatten()),
                                  xc[done].flatten(),
                                  (grid[...,0].flatten(), grid[...,1].flatten()),
                                  method='cubic', fill_value=np.nan)
        linear_yc      = griddata((grid[...,0][done].flatten(), grid[...,1][done].flatten()),
                                  yc[done].flatten(),
                                  (grid[...,0].flatten(), grid[...,1].flatten()),
                                  method='cubic', fill_value=np.nan)
        
        linear_stddev = np.reshape(linear_stddev, (self.lut_dim, self.lut_dim))
        linear_xc     = np.reshape(linear_xc, (self.lut_dim, self.lut_dim))
        linear_yc     = np.reshape(linear_yc, (self.lut_dim, self.lut_dim))

        # currently restrict linear/cubic to the area around the wrap arround point to
        # prevent strange effects around the convex hull edge
        if self.uv_mode:
            u_min = int(0.17 * self.lut_dim)
            u_max = int(0.27 * self.lut_dim)
            v_min = int(0.06 * self.lut_dim)
            v_max = int(0.50 * self.lut_dim)
            linear_stddev[:u_min, :] = np.nan
            linear_stddev[u_max:, :] = np.nan
            linear_stddev[:, :v_min] = np.nan
            linear_stddev[:, v_max:] = np.nan
            linear_xc[:u_min, :]     = np.nan
            linear_xc[u_max:, :]     = np.nan
            linear_xc[:, :v_min]     = np.nan
            linear_xc[:, v_max:]     = np.nan
            linear_yc[:u_min, :]     = np.nan
            linear_yc[u_max:, :]     = np.nan
            linear_yc[:, :v_min]     = np.nan
            linear_yc[:, v_max:]     = np.nan

        xc[~done]     = linear_xc[~done]
        yc[~done]     = linear_yc[~done]
        stddev[~done] = linear_stddev[~done]
        done = ~np.isnan(xc)

        nearest_stddev = griddata((grid[...,0][done].flatten(), grid[...,1][done].flatten()),
                                  stddev[done].flatten(),
                                  (grid[...,0].flatten(), grid[...,1].flatten()),
                                  method='nearest')
        nearest_xc     = griddata((grid[...,0][done].flatten(), grid[...,1][done].flatten()),
                                  xc[done].flatten(),
                                  (grid[...,0].flatten(), grid[...,1].flatten()),
                                  method='nearest')
        nearest_yc     = griddata((grid[...,0][done].flatten(), grid[...,1][done].flatten()),
                                  yc[done].flatten(),
                                  (grid[...,0].flatten(), grid[...,1].flatten()),
                                  method='nearest')

        nearest_stddev = np.reshape(nearest_stddev, (self.lut_dim, self.lut_dim))
        nearest_xc     = np.reshape(nearest_xc, (self.lut_dim, self.lut_dim))
        nearest_yc     = np.reshape(nearest_yc, (self.lut_dim, self.lut_dim))
        
        xc[~done]     = nearest_xc[~done]
        yc[~done]     = nearest_yc[~done]
        stddev[~done] = nearest_stddev[~done]

        tmp = self.arcToPeak(np.array([xc.flatten(), yc.flatten()]).T)
        self.fit_grid[...,0] = np.reshape(tmp, (self.lut_dim, self.lut_dim))
        self.fit_grid[...,1] = stddev
        self.fit_done[:,:]   = True

        return True

    
    def lutLookup(self, uv = None, xy = None):
        """
        Interpolate values from LUT.
        Last dimension is uv/xy selector
        """

        lookup = None

        if uv is None and xy is None:
            raise Exception('Give either an array of uv or xy values')

        if self.uv_mode and not uv is None:
            lookup = uv.copy()
        elif self.uv_mode and uv is None:
            lookup = KSM.xyToUv(xy)
        elif not uv is None:
            lookup = KSM.uvToXy(uv)
        else:
            lookup = xy.copy()

        max_idx = self.lut_dim - 1

        lookup = lookup * max_idx
        lu_int = lookup.astype(int)
        lu_int[lu_int < 0] = 0
        lu_int[lu_int > max_idx] = max_idx
        weight_upper = lookup - lu_int
        weight_lower = 1.0 - weight_upper

        griddata = np.zeros((self.lut_dim, self.lut_dim, 3))
        griddata[...,2]  = self.fit_grid[...,1]

        xc,yc = self.peakToArc(self.fit_grid[...,0])
        tmp = np.array([xc.flatten(), yc.flatten()]).T
        griddata[...,:2] = np.reshape(tmp, (self.lut_dim, self.lut_dim, 2))

        result    = np.zeros(lookup.shape[:-1] + (3,))

        for i in range(3):
            result[...,i]  = griddata[...,i][lu_int[...,0], lu_int[...,1]] * weight_lower[...,0]
            result[...,i] += griddata[...,i][np.minimum(lu_int[...,0] + 1, max_idx), lu_int[...,1]] * weight_upper[...,0]
            result[...,i] *= weight_lower[...,1]
            result[...,i] += griddata[...,i][lu_int[...,0], np.minimum(lu_int[...,1] + 1, max_idx)] * weight_lower[...,0] * weight_upper[...,1]
            result[...,i] += griddata[...,i][np.minimum(lu_int[...,0] + 1, max_idx), np.minimum(lu_int[...,1] + 1, max_idx)] * weight_upper[...,0] * weight_upper[...,1]
        
        res = np.zeros(lookup.shape)
        res[...,0] = self.arcToPeak(result[...,:2])
        res[...,1] = result[...,2]
        
        return res
    
    def evalImageSaturation(self, xyz, saturation, luma_preserve=0.0):
        """
        xyz = XYZ (CIE-E)
        saturation = [0, inf)
        """
        eps = 0.000000001

        if len(xyz.shape) == 1:
            xyz = xyz[None,...].copy()

        s = np.sum(xyz, axis=-1)
        idx_black = s < eps
        s[idx_black] = 1.0
        s = np.repeat(s[...,None], 2, axis=-1)
        xy = xyz[...,:2] / s
        xy[idx_black] = 1.0/3.0
        gauss_data = self.lutLookup(xy=xy)
        s = None
        xy = None

        # calculate reference multiplier
        # gvals = self.gaussNp(gauss_data)
        y = xyz[...,1]
        idx_black = np.logical_or(idx_black, y < eps)
        y[idx_black] = 1
        y_tmp = self.evalSpecXyz(gparams=gauss_data)[...,1]
        idx_black = np.logical_or(idx_black, y_tmp < eps)
        y_tmp[idx_black] = 1
        mult  = y / y_tmp
        y_tmp = None

        # apply saturation
        gauss_data[...,1] = KSM.thetaToStddev(KSM.stddevToTheta(gauss_data[...,1]) * saturation)
        # gvals = self.gaussNp(gparams=gauss_data)
        xyz_ref = self.evalSpecXyz(gparams=gauss_data) * np.repeat(mult[...,None], (3), axis=-1)
        invalid_pixels = np.repeat((np.sum(np.isnan(xyz_ref), axis=-1) > 1)[...,None], (3), axis=-1)
        xyz_ref[invalid_pixels] = eps
        # invalid_pixels = None

        y_new = xyz_ref[...,1]
        idx_black = np.logical_or(idx_black, y_new < eps)
        y_new[idx_black] = 1
        y_mult  = y / y_new
        y_new = None
        y = None

        xyz_ref_cal = xyz_ref * np.repeat(y_mult[...,None], (3), axis=-1)
        xyz_ref_cal[invalid_pixels] = eps
        invalid_pixels = None

        # calculate final xyz
        res = xyz_ref * (1 - luma_preserve) + xyz_ref_cal * luma_preserve
        res[idx_black] = eps
        return res


    def stddevToTheta(stddev):
        eps = 0.0000001
        if isinstance(stddev, int) or isinstance(stddev, float):
            stddev_copy = np.array([stddev], dtype=np.float64)
        else:
            stddev_copy = np.array(stddev, dtype=np.float64)
        stddev_copy[stddev_copy < eps] = eps
        return 1.0 / np.power(stddev_copy, 2)
    
    def thetaToStddev(theta):
        eps = 0.0000001
        if isinstance(theta, int) or isinstance(theta, float):
            theta_copy = np.array([theta], dtype=np.float64)
        else:
            theta_copy = np.array(theta, dtype=np.float64)
        theta_copy[theta_copy < eps] = eps
        return np.sqrt(1.0 / theta_copy)