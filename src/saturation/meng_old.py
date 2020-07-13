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
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay

# Colour is used for plotting of spectra and calculation of standard illuminants
import colour
from colour.plotting import *
import colour.plotting
import colour.colorimetry as colorimetry

# Our helper for loading spectra/interpolating
from .spectra_loader import read_spectra, interpolate, read_cie1931


class MinimizingUpsampler:
    """
    Class offering upsampling of spectra for a given set of tristimuli and a given
    spectral sensitivity like XYZ or camera RGB tristimuli

    The code for fitting the spectrum using python multiprocessing was derived from
    http://cg.ivd.kit.edu/spectrum.php
    where the supplemental material of
    ``Physically Meaningful Rendering using Tristimulus Colours'' by Johannes Meng,
    Florian Simon, Johannes Hanika and Carsten Dachsbacher from the KIT Karlsruhe,
    published in Computer Graphics Forum (Proceedings of EGSR 2015)
    has been published which is the base for this spectral upsampling model extended
    by a saturation operation and simpliefied for faster lookup and flexibility
    regarding the used color matching functions.
    """

    def __init__(self):
        """
        Initialize class and create instance variables
        """
        self.response = np.array([])
        self.xyz = None
        self.convex_hull = None
        self.delaune = None
        self.points_to_solve = []
        self.lut_dim = 0
        self.lut_dim_mult = 0
        self.lut_grid = None
        self.lut_done = None

        self.maxiter = 2000
        self.ftol = 1e-10
        self.rgbsum = 1

        self.spectra_epsilon = 1e-6
        self.threads = os.cpu_count()

        self.fit_constraint = None
        self.fit_objective = None
        self.fit_bounds = None
        self.fit_x0 = None
        self.fit_gain = None
        self.gaus_params = None

    def setResponse(self, response):
        """
        Set camera response. Remove lowest part until first non-zero value.
        Also remove upper area from the last non-zero entry. Camera response
        is cut to prevent meaningless values in fit for wavelengths
        where the incident spectrum has no influence on the tristimuli.

        self.xyz is also initialized with the CIE-XYZ 1931 curve for the same
        range of wavelengths.
        """
        if response.shape[0] > 40 and response.shape[1] == 4:
            low = 0
            high = response.shape[0] - 1
            for i in range(0, response.shape[0]):
                if np.max(response[i, 1:]) > self.spectra_epsilon:
                    low = i
                    break
            for i in range(response.shape[0] - 1, 0, -1):
                if np.max(response[i, 1:]) > self.spectra_epsilon:
                    high = i
                    break
            self.response = response[low:high+1, :].copy()
            # ciefile = os.path.join(os.path.dirname(__file__), '../../data/xyz_1nm_360_830.csv')
            xyz = read_cie1931()
            self.xyz = self.reduceToResponseRange(xyz)
        else:
            return False


    def fit(self, rgb):
        """
        Do the fit (minimization via Sequential Least SQuares Programming)
        for a given rgb tripple to get a meaningfull spectrum that fullfils
        the smoothness condition.
        """
        rgb  = np.array(rgb)
        x0   = self.fit_x0.copy()
        bnds = self.fit_bounds

        objective = lambda x: self.fit_objective(x)
        cnstr = None
        if not self.fit_constraint is None:
            cnstr = {
                'type': 'eq',
                'fun': lambda s: (np.array(self.fit_constraint(s)) - rgb)
            }

        res = minimize(objective, x0, method='SLSQP', constraints=cnstr,
                       bounds=bnds, options={"maxiter": self.maxiter, "ftol": self.ftol})

        if not res.success:
            err_message = 'Error for xyz={} after {} iterations: {}'.format(rgb, res.nit, res.message)
            return ([0] * num_bins, True, err_message)
        else:
            gain = self.fit_gain(res)
            # The result may contain some very tiny negative values due
            # to numerical issues. Clamp those to 0.
            #print('Fit result for: ', rgb, '  ', res.x)
            return ([max(x, 0) * gain for x in res.x], False, "")

    def fitPq(self, pq, rgbsum = -1):
        """
        Do a fit for a given pq coordinate. Assumes an overall sum for base 'brightness'.
        """
        if rgbsum <= 0:
            rgbsum = self.rgbsum
        r = pq[0] * rgbsum
        g = pq[1] * rgbsum
        return self.fit([r, g, rgbsum - r - g])

    def calculateLasers(self):
        """
        Calculate a set of laser like spectra matching the sensor response
        wavelength range.
        """
        rgb = []
        spectrum = np.zeros(self.response.shape[0])
        for i in range(0, spectrum.shape[0]):
            spectrum[:] = 0
            spectrum[i] = 1
            rgb.append(self.evalSpectrum(spectrum))
        return np.array(rgb)

    def pqLasers(self):
        """
        Build laser representation in pq space
        """
        lasers = self.calculateLasers()
        pq = np.zeros((lasers.shape[0], 2))
        pq[:,0] = lasers[:,0] / np.sum(lasers, axis=1)
        pq[:,1] = lasers[:,1] / np.sum(lasers, axis=1)
        return pq

    def rgbNormalize(self, spec, rgbsum = -1):
        """
        Normalize the spectrum to have a sum of rgbsum after evaluation.
        """
        if rgbsum <= 0:
            rgbsum = self.rgbsum
        rgb = np.array(self.evalSpectrum(spec))
        s = np.sum(rgb)
        if s > 0:
            return spec * (rgbsum / s)
        return False

    def buildHull(self):
        """
        Build convex hull of valid spectra for given camera response
        """
        pq = self.pqLasers()
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

    def estimateTriangleSize(self, lut_dimension):
        """
        Estimate number of points to solve for the current convex hull for given
        LUT dimension.

        Multi threading inspired by the lut creation code in the Meng 2015 paper.
        """
        self.points_to_solve = []
        self.lut_dim = lut_dimension
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

    def gaussianPrep(self):
        """
        Prepare helper arrays for gauss calculation.
        """
        self.gaus_params = {}
        size = self.response[-1,0] - self.response[0,0] + 1
        self.gaus_params['inrange_up'] = np.linspace(0, self.response[-1,0] - self.response[0,0], num=size)
        self.gaus_params['inrange_down'] = np.absolute(self.gaus_params['inrange_up'] - (self.response[-1,0]-self.response[0,0]))
        self.gaus_params['inrange_wl'] = np.linspace(self.response[0,0], self.response[-1,0], num=size)

    def gauss(self, peak, maximum, sig):
        """
        Calculate gaussian distribution with a given peak, maximum and sigma
        """
        peak_to_end = self.response[-1,0] - peak
        peak_to_front = peak - self.response[0,0]
        tmp = np.absolute(self.gaus_params['inrange_up'] - peak_to_front)

        if peak_to_front > peak_to_end:
            tmp = np.minimum(tmp, peak_to_end + self.gaus_params['inrange_up'])
        else:
            tmp = np.minimum(tmp, peak_to_front + self.gaus_params['inrange_down'])

        return np.exp(-np.power(tmp, 2.) / (10 * np.power(sig*sig, 2.))) * maximum

    def setupFitterModel(self, model = 'meng'):
        """
        Setup fitter model parameters
        """
        num_bins = self.response.shape[0]
        self.fit_bounds = [(0, 1000)] * num_bins
        self.fit_x0 = [1] * num_bins
        self.fit_objective = lambda s: sum([(s[i] - s[i+1])**2 for i in range(len(s)-1)])
        self.fit_constraint = lambda s: self.evalSpectrum(s)
        self.fit_gain = lambda r: self.response[1, 0] - self.response[0, 0]

        if model == 'ksm':
            print('KSM mode')
            self.gaussianPrep()
            num_bins = 3
            self.fit_bounds = [(self.response[0, 0], self.response[-1, 0]),
                               (0, 3),
                               (0.001, 20.0)]
            self.fit_x0 = [ 550, 0.01, 5 ]
            self.fit_objective = lambda s: 1 / (s[2]**2)
            self.fit_constraint = lambda s: self.evalSpectrum(self.gauss(s[0], s[1], s[2]))
            # self.fit_constraint = lambda s: 1 / (s[2]**2)
            # self.fit_objective = lambda s: self.evalSpectrum(self.gauss(s[0], s[1], s[2]))
            self.fit_gain = lambda x: 1
        else:
            print('regular spectra fitting')

        return num_bins


    def fillInnerLut(self, model = 'meng'):
        """
        Fill all point inside the LUT that result in a valid minimization.

        This step takes long depending on the LUT dimensions

        model parameter is either KSM for KSM kernel fitting (or currently meng style fit
        for every other value)
        """

        # clear LUT and Grid first
        self.lut_grid = None
        self.lut_done = None

        num_bins = self.setupFitterModel(model)

        grid = np.ctypeslib.as_ctypes(np.zeros((self.lut_dim, self.lut_dim, num_bins)))
        grid_shared = sharedctypes.RawArray(grid._type_, grid)
        done = np.ctypeslib.as_ctypes(np.full((self.lut_dim, self.lut_dim), False, dtype=bool))
        done_shared = sharedctypes.RawArray(done._type_, done)

        num_procs = self.threads-1

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
                        fit_res = self.fitPq([xx, yy])
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

    def fillMissing(self):
        """
        Fill missing points by using nearest neighboor in hull for points outside convex hull
        and after that check for any missing point left and fill with interpolation of surrounding.
        """

        # Helper to calculate distance of points
        def dist(ref, test):
            x = ref[0] - test[0]
            y = ref[1] - test[1]
            return math.sqrt(x*x + y*y)

        grid = self.lut_grid
        done = self.lut_done
        done_ref = self.lut_done.copy()

        vector = np.array([list(range(0, grid.shape[0]))])
        grid_size = np.stack((matlib.repmat(vector.transpose(), 1, grid.shape[0]), matlib.repmat(vector, grid.shape[0], 1)), axis=2)
        grid_float = grid_size / (grid.shape[0] - 1.0)
        grid_min = grid_size[done]
        grid_min_float = grid_min / (grid.shape[0] - 1.0)

        midpoint = [1.0/3, 1.0/3]
        inside_missing = np.full((grid.shape[0], grid.shape[1]), False, dtype=bool)

        # At first interpolate areas outside the convex hull
        for y in range(0, done.shape[1]):
            for x in range(0, done.shape[0]):
                if not done_ref[x, y]:
                    xf = x / (grid.shape[0] - 1)
                    yf = y / (grid.shape[1] - 1)
                    if self.inHull([xf, yf]):
                        inside_missing[x, y] = True
                        continue
                    delta = grid_min_float - np.asarray([xf, yf])
                    dist_2 = np.sum(delta**2, axis=1)
                    nearest_grid = np.unravel_index(np.argmin(dist_2), grid_min_float.shape[0])
                    pos = grid_min[nearest_grid]
                    grid[x, y, :] = grid[pos[0], pos[1], :]
                    done[x, y] = True

        # Interpolate last missing entries, in case some inside entries where missing
        # due to minimizer not working correctly. Just interpolats with surrounding etnries
        for el in grid_size[inside_missing]:
            spec = np.zeros(grid.shape[2])
            weight = 0
            for x in range(el[0] - 1, el[0] + 2):
                if x >= 0 and x < grid.shape[0]:
                    for y in range(el[1] - 1, el[1] + 2):
                        if x != y and not inside_missing[x, y] and y >= 0 and y < grid.shape[1]:
                            w = 1.0 / dist(el, [x, y])
                            weight += w
                            spec += grid[x, y, :]
            if weight > 0:
                grid[el[0], el[1], :] = spec / weight
                inside_missing[el[0], el[1]] = False
            else:
                print('Error in sample: ', el)

    def evalSpectrum(self, spectrum):
        """
        Evaluate spectrum against the current minimizer instance response curve
        """
        return np.array([np.dot(spectrum, self.response[:,i]) for i in range(1, 4)])

    def evalXYZ(self, spectrum):
        """
        Evaluate spectrum against XYZ curve
        """
        return np.array([np.dot(spectrum, self.xyz[:,i]) for i in range(1, 4)])

    def evalSpectrumWithResponse(self, spectrum, response):
        """
        Evaluate spectrum against the current minimizer instance response curve
        """
        return np.array([np.dot(spectrum, response[:,i]) for i in range(1, 4)])

    def visualizeLut(self, dimensions):
        """
        Visualize the LUT as XYZ.
        """
        base = np.array([range(dimensions)])
        hor  = np.matlib.repmat(base, dimensions, 1)
        ver  = np.matlib.repmat(base.transpose(), 1, dimensions)
        # Todo: Improve scaling/clipping of input values
        # scale with dimensions - 0.999999 to prevent points on the edge
        stacked = np.stack((hor, ver), axis=2) / (dimensions - 0.9999)
        return self.interpolateImage(stacked)

    def interpolateImage(self, img, as_xyz = True, illum = None, ref_illum = None):
        """
        Interpolate an image through the LUT.

        as_xyz: If true, evaluate as XYZ. If false, result will be spectral
        """
        #check dimensions
        bkp = None
        dims = 3
        if as_xyz:
            bkp = self.response
            self.response = self.xyz
        else:
            dims = self.response.shape[0]
        interpolated = np.zeros((img.shape[0], img.shape[1], dims))

        pq = None
        sum_arr = None
        # todo: prevent division by zeros here!
        # todo: preven 1.0 values
        if img.shape[2] == 2:
            pq = img
        else:
            sum_arr = np.sum(img, axis=2)
            pq = np.stack((img[:,:,0] / sum_arr, img[:,:,1] / sum_arr), axis=2)

        # Todo Vectorize at least parts of it?
        for y in range(pq.shape[0]):
            for x in range(pq.shape[1]):
                interpolated[y, x, :] = self.gridInterpolate(pq[y, x, :], eval = as_xyz, illum = illum, ref_illum = ref_illum)
                if sum_arr is not None:
                    interpolated[y, x, :] *= sum_arr[y, x]

        if as_xyz:
            self.response = bkp
        return interpolated

    def gridInterpolate(self, pq, eval = False, illum = None, ref_illum = None):
        """
        Interpolate point in grid.
        """

        # todo prevent out of range
        x = pq[0] * (self.lut_grid.shape[0] - 1)
        y = pq[1] * (self.lut_grid.shape[0] - 1)
        xl = math.floor(x)
        yl = math.floor(y)
        weight_x = xl + 1 - x
        weight_y = yl + 1 - y
        top = self.lut_grid[xl, yl+1] * weight_x + self.lut_grid[xl+1, yl+1] * (1 - weight_x)
        bot = self.lut_grid[xl, yl+0] * weight_x + self.lut_grid[xl+1, yl+0] * (1 - weight_x)
        combined = bot * weight_y + top * (1 - weight_y)

        if illum is not None:
            combined = combined / illum
        if ref_illum is not None:
            combined = combined * ref_illum

        if eval:
            combined = np.array(self.evalSpectrum(combined))
        return combined

    def load(lut_folder, intermediate = False):
        """
        Load intermediate or final LUT for processing images
        """
        if lut_folder[-1] != '/':
            lut_folder += '/'
        grid_filename = lut_folder + 'lut_full.npy'

        up = MinimizingUpsampler()

        for file in os.listdir(lut_folder):
            if file.endswith('.used.csv'):
                up.setResponse(read_spectra(os.path.join(lut_folder, file)))

        if intermediate:
            grid_filename = lut_folder + 'gridfile.npy'
            up.lut_done = np.load(lut_folder + 'donefile.npy')

        up.lut_grid = np.load(grid_filename)
        up.lut_dim = up.lut_grid.shape[0]
        up.lut_dim_mult = 1.0 / (up.lut_dim - 1)

        if intermediate:
            up.buildHull()
            up.estimateTriangleSize(up.lut_dim)

        model = 'meng'
        if up.lut_grid.shape[2] == 3:
            model = 'ksm'
        up.setupFitterModel(model)

        return up

    def getIlluminantInRange(self, spd_description):
        """
        Load an illuminant and scale to camera spectra range. Scale for equal
        luminance as well to have roughly consistent Y values when applying
        illuminant correction and white point adaption.
        """

        return self.getIlluminant(spd_description)

    def reduceToResponseRange(self, spec):
        """
        Calculate spectrum to fit the range of the response.
        """
        result = np.zeros((self.response.shape[0], spec.shape[1]))
        result[:,0] = self.response[:,0]
        start = self.response[0, 0] - spec[0, 0]
        # Todo check for negative start -> Meaning spec is to short
        last = result.shape[0]
        if spec.shape[0] - start < last:
            last = int(spec.shape[0] - start + 0.0001)
        result[:last, 1:] = spec[int(start+0.0001):int(start+self.response.shape[0]+0.0001), 1:]
        return result

    def getIlluminant(self, illuminant_name):
        """
        Load an illuminant and scale to camera spectra range. Scale for equal
        luminance as well to have roughly consistent Y values when applying
        illuminant correction and white point adaption.

        Valid inputs are D**** (D6500 etc.) for Daylight illuminants
        and ****K (3050K etc.) for blackbody radiation
        """

        illuminant = None

        # Decide if using Dayling spectra or Tungsten based on input string
        if illuminant_name[0] == 'D':
            cct_d = float(illuminant_name[1:]) * 1.4388 / 1.4380
            xy = colour.temperature.CCT_to_xy_CIE_D(cct_d)
            illuminant = colorimetry.sd_CIE_illuminant_D_series(xy)
        elif illuminant_name[-1] == 'K':
            shape = colorimetry.DEFAULT_SPECTRAL_SHAPE
            shape.start = 300
            shape.end = 830
            shape.interval = 1
            illuminant = colorimetry.sd_blackbody(float(illuminant_name[:-1]), shape=shape)
        else:
            raise NameError('Unknown illuminant requested')
        data = np.array([illuminant.wavelengths, illuminant.values]).transpose()
        interp = interpolate(data)
        illum = self.reduceToResponseRange(interp)

        luminance = np.dot(illum[:,1], self.xyz[:,2])
        illum[:,1] = illum[:,1] / luminance
        return illum

    def getSmoothIlluminant(self, illuminant_name):
        """
        Calculate a smoothed variant of a standard illuminant by getting the
        standard illuminat.

        This method is slow as a full minimizer fit is executed to find a smooth
        Meng et al. based spectrum for the given illuminant.
        """

        illum = self.getIlluminant(illuminant_name)
        xyz = self.evalXYZ(illum[:,1])
        response = self.response
        self.response = self.xyz
        fit = self.fit(xyz)
        self.response = response
        if fit[1]:
            return None
        fit = np.stack((illum[:,0], np.array(fit[0]))).transpose()
        return fit


    def getXyzGrid(self, illum = None, ref_illum = None):
        """
        Convert spectral grid to XYZ grid.
        """

        res = np.zeros((self.lut_grid.shape[0], self.lut_grid.shape[1], 3))

        for p in range(self.lut_grid.shape[0]):
            for q in range(self.lut_grid.shape[1]):
                combined = self.lut_grid[p, q]

                if illum is not None:
                    combined = combined / illum
                if ref_illum is not None:
                    combined = combined * ref_illum

                res[p, q, :] = np.array(self.evalXYZ(combined))

        return res

    def subsampleBin(self, wavelengths, spec, bin_size = 5, start_wl = 380, end_wl = 780):
        numbins = (end_wl - start_wl) // bin_size
        in_len = wavelengths.shape[0]

        start_idx = np.argwhere(np.abs(wavelengths - start_wl) < 0.001)
        if len(start_idx) == 0:
            start_idx = int(-(wavelengths[0] - start_wl))
        else:
            start_idx = start_idx[0][0]

        weights = np.ones((bin_size + 1))
        weights[0]  = 0.5
        weights[-1] = 0.5
        
        result = np.zeros((numbins))
        for i in range(numbins):
            vtmp = 0
            wtmp = 0
            idx = start_idx + i * bin_size
            for j in range(len(weights)):
                if (idx + j) < in_len and (idx + j) >= 0:
                    vtmp += weights[j] * spec[idx + j]
                    wtmp += weights[j]
            if wtmp > 0:
                result[i] = vtmp  # disable division to keep  / wtmp
    
        return result