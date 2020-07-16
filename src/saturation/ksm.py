from .optimizerbase import *

class KSM(OptimizerBase):

    def __init__(self, start_wl = 380, end_wl = 780, *args, **kwargs):
        self._TYPE = 'KSM'
        super().__init__(*args, **kwargs)

        self.xyz        = reduce_to_wavelengths(self.xyz, start_wl, end_wl)
        self.maxiter    = 10000  # Check for sane values
        self.ftol       = 1e-8

        self._max_dist  = 0.07
        self._theta_min = 1e-7
        self._theta_max = 25

        self._scatter   = None
        self._indices   = None

        self._prepRangesCache()

    def _prepRangesCache(self):
        self.wlrange    = self.xyz[:,0]
        self.wlin       = self.wlrange.copy()
        self.wlin_u     = self.wlin[-1]
        self.wlin_l     = self.wlin[0]
        self.wlin_half  = (self.wlin_u - self.wlin_l) // 2

        self.inrange0   = self.wlin - self.wlin_l
        self.inrange0_u = self.inrange0[-1]
        self.inrange1   = self.wlin_u - self.wlin_l - self.inrange0
    
    def save(self, folder, *args, **kwargs):
        ksm_struct = {
            'ksm_max_dist':  self._max_dist,
            'ksm_theta_min': self._theta_min,
            'ksm_theta_max': self._theta_max,
        }
        if self._saveBase(folder, ksm_struct, *args, **kwargs):
            if not self._scatter is None:
                np.savez_compressed(os.path.join(folder, 'ksm_scatter.npz'), self._scatter)
            return True
        
        print('Failed to save')
        return False

    def load(folder):
        obj = Meng(None)
        data = obj._loadBase(folder)
        if data:
            if os.path.exists(os.path.join(folder, 'ksm_scatter.npz')):
                obj._scatter = np.load(os.path.join(folder, 'ksm_scatter.npz'))['arr_0']
                i = obj._scatter.shape[0]
                obj._indices = np.linspace(0, i-1, num=i, dtype=np.uint32)
            else:
                obj.fitter_state = FitterState.INIT

            if 'ksm_max_dist' in data:
                obj._max_dist = data['ksm_max_dist']
            if 'ksm_theta_min' in data:
                obj._theta_min = data['ksm_theta_min']
            if 'ksm_theta_max' in data:
                obj._theta_max = data['ksm_theta_max']

            obj._prepRangesCache()
            return obj
        return None

    def gauss(self, params):
        peak = params[...,0] - self.wlin_l
        peak = np.repeat(peak[..., None], self.inrange1.shape[-1], axis=-1)
        theta = params[...,1]
        theta = np.repeat(theta[..., None], self.inrange1.shape[-1], axis=-1)
        vrange = np.absolute(peak - self.inrange0)
        idx = peak < self.wlin_half
        wrange = np.zeros(vrange.shape)
        if np.sum(idx) > 0:
            wrange[idx]  = peak[idx] + np.tile(self.inrange1, (idx[idx].shape[0] // self.inrange1.shape[0],))
        if np.sum(~idx) > 0:
            wrange[~idx] = np.absolute(-peak[~idx] + np.tile(self.inrange0 + self.inrange0_u, (idx[~idx].shape[0] // self.inrange1.shape[0],)))
        mrange = np.minimum(vrange, wrange)
        return np.exp(-np.multiply(np.power(mrange, 2.), theta))

    def _prepareOptimizer(self):
        # build scatter
        start_wl     = self.wlin[0]
        end_wl       = self.wlin[-1]
        size         = int(end_wl-start_wl + 1.0001)
        inrange_up   = np.linspace(0, end_wl - start_wl, num=size)
        inrange_down = np.absolute(inrange_up - (end_wl-start_wl))
        inrange_wl   = np.linspace(start_wl, end_wl, num=size)
        steps_wl     = lambda x: size + (size - 1) * (x-1)
        steps_sig    = 80

        peak_space   = np.linspace(start_wl, end_wl, num=steps_wl(3))
        sig_space    = np.linspace(0.01, 30, num=steps_sig)
        k_space      = [1, 1.35, 1.7, 2]

        # 2nd dimension is x(p), y(q), peak, theta
        gauss_funcs = np.zeros((peak_space.shape[0] * sig_space.shape[0] * len(k_space), 4))
        maximum     = 1.0

        i   = 0
        cnt = 0
        # sig_max = 0
        for peak in peak_space:
            for sig_val in sig_space:
                for k in k_space:
                    sig     = sig_val**k  #**2
                    #sig_max = max(sig_max, sig)
                    theta   = KSM.stddevToTheta(np.array([sig]))[0]
                    val     = self.gauss(np.array([peak, theta]))
                    xy      = OptimizerBase.xyzToXy(self.toXyz(val[None, ...]))

                    if not np.isnan(xy[0,0]) and not np.isnan(xy[0,1]):
                        gauss_funcs[i, 0:2] = xy[0,:]
                        gauss_funcs[i, 2]   = peak
                        gauss_funcs[i, 3]   = theta
                        i += 1
                    else:
                        cnt += 1
        print("_prepareOptimizer invalid entries: ", cnt)

        self._scatter = gauss_funcs[0:i, :].copy()
        self._indices = np.linspace(0, i-1, num=i, dtype=np.uint32)

        self.__fit_bounds  = [(self.wlin_l, self.wlin_u), (self._theta_min, self._theta_max)]
        self.fitter_state  = FitterState.READY
    
    def getX0(self, xy):
        max_dist = self._max_dist
        mdist = 0.0
        dist  = np.ones((10))
        while np.sum(dist < mdist) < 1 and mdist < max_dist:
            mdist += 0.005
            dist   = np.sqrt(np.power(self._scatter[:,0] - xy[0], 2) + np.power(self._scatter[:,1] - xy[1], 2))
        if mdist >= max_dist:
            return np.array([555, 0.0001])

        idx  = self._indices[dist < mdist]
        midx = np.argmin(self._scatter[idx, 3])
        return self._scatter[idx[midx], 2:]
    
    def fit(self, default = None, xy = None, xyz = None, uv = None):
        if self.fitter_state.value < FitterState.READY.value:
            return False

        # Prepare input values
        pos = None
        txy = None
        if not default is None:
            if self.is_uv:
                txy = OptimizerBase.uvToXy(np.array(default))
            else:
                txy = default
            pos = default
        elif not xy is None:
            if self.is_uv:
                pos = OptimizerBase.xyToUv(np.array(xy))
            else:
                pos = xy
            txy = xy
        elif not uv is None:
            if self.is_uv:
                pos = uv
            else:
                pos = OptimizerBase.uvToXy(np.array(uv))
            txy = OptimizerBase.uvToXy(np.array(uv))
        elif not xyz is None:
            txy = OptimizerBase.xyzToXy(np.array(xyz))
            if self.is_uv:
                pos = OptimizerBase.xyToUv(txy)
            else:
                pos = txy
        else:
            print('Invalid value call')
            return ([0] * self.lut_entry_values, True, 'Invalid input!')

        
        # prepare fitter

        def objective(s):
            # Target is to get Theta (s[1]) as low as possible -> smooth spectra
            return s[1]

        def constrHelper(x):
            params = np.array(x)
            xyz    = spectra_integrate(self.gauss(params[None, ...]), self.xyz[:, 1:])
            return OptimizerBase.xyzToXy(xyz).flatten() - txy

        x0        = self.getX0(txy)
        bounds    = self.__fit_bounds.copy()
        ftol      = self.ftol

        cnstr     = {
                'type': 'eq',
                'fun': constrHelper
            }
        res = minimize(objective, x0, method='SLSQP', constraints=cnstr,
                       bounds=bounds, options={"maxiter": self.maxiter, "ftol": ftol})
        
        if not res.success:
            err_message = 'Error for pos={} after {} iterations: {}'.format(pos, res.nit, res.message)
            return ([0] * 3, True, err_message)
        else:
            rval     = np.zeros((3))
            rval[:2] = self.peakToArc(np.array(res.x[0]))
            rval[2]  = res.x[1]
            return (rval, False, "")
    
    # Todo test if old cubic interpolation is advicable
    #def postOptimizer(self):
    #    print('No real work done in post process, needs conversion to coordinates instead of ')
    
    def lookup(self, **kwargs):
        pq, s       = self.toCoordinates(**kwargs)
        lookup      = self.bilinearInterpolation(pq)
        data        = np.ones(lookup.shape)
        data[...,1] = self.arcToPeak(lookup[...,:2])
        data[...,2] = lookup[...,2]
        if s is None:
            return data[...,1:]
        
        y = kwargs['xyz'][...,1].copy()
        rec = self.toXyz(data)
        data[...,0] = np.divide(y, rec[...,1])
        return data

    def saturate(self, data, saturation):
        res        = data.copy()
        res[...,2] = res[...,2] * saturation
        return res

    def toXyz(self, data):
        xyz = spectra_integrate(self.gauss(data[...,1:]), self.xyz[:,1:])
        return np.multiply(xyz, data[...,0][...,None])

    def peakToArc(self, peak_wl):
        """
        Convert peak wavelength value to circle arc position as xy coordinates
        """
        res        = np.zeros(peak_wl.shape + (2,))
        multiplier = 2 * np.pi / (self.wlin_u - self.wlin_l)
        alpha      = (peak_wl - self.wlin_l) * multiplier
        res[...,0] = np.cos(alpha)
        res[...,1] = np.sin(alpha)
        return res

    def arcToPeak(self, coords):
        """
        Convert peak wavelength value to circle arc position as xy coordinates
        """
        multiplier = 2 * np.pi / (self.wlin_u - self.wlin_l)
        alpha      = np.array(np.arctan2(coords[..., 1], coords[..., 0]))
        alpha[alpha < 0] = 2 * np.pi + alpha[alpha < 0]
        while (np.any(alpha > 2 * np.pi)):
            alpha[alpha > 2 * np.pi] = alpha[alpha > 2 * np.pi] - 2*np.pi
        return (alpha / multiplier) + self.wlin_l


    def stddevToTheta(stddev):
        eps = 1e-8
        if isinstance(stddev, int) or isinstance(stddev, float):
            stddev_copy = np.array([stddev], dtype=np.float64)
        else:
            stddev_copy = np.array(stddev, dtype=np.float64)
        stddev_copy[stddev_copy < eps] = eps
        return 1.0 / np.power(stddev_copy, 2)
    
    def thetaToStddev(theta):
        eps = 1e-8
        if isinstance(theta, int) or isinstance(theta, float):
            theta_copy = np.array([theta], dtype=np.float64)
        else:
            theta_copy = np.array(theta, dtype=np.float64)
        theta_copy[theta_copy < eps] = eps
        return np.sqrt(1.0 / theta_copy)