from .optimizerbase import *

class Meng(OptimizerBase):

    def __init__(self, response, *args, **kwargs):
        self._TYPE = 'Meng'
        if not 'is_uv' in kwargs:
            kwargs['is_uv'] = False
        super().__init__(*args, **kwargs)
        if isinstance(response, str) :
            self.response = reduce_spectra(read_spectra(response))[0]
            self.xyz      = reduce_to_wavelengths(self.xyz, self.response[0, 0], self.response[-1, 0])
        elif isinstance(response, np.ndarray):
            self.response = response
            self.xyz      = reduce_to_wavelengths(self.xyz, self.response[0, 0], self.response[-1, 0])

    def save(self, folder, *args, **kwargs):
        meng_struct = {}
        if self._saveBase(folder, meng_struct, *args, **kwargs):
            return True
        
        print('Failed to save')
        return False

    def load(folder):
        obj = Meng(None)
        data = obj._loadBase(folder)
        if data:
            return obj
        return None

    def _prepareOptimizer(self):
        self.lut_entry_values = self.response.shape[0]
        self.__fit_x0         = [1] * self.lut_entry_values
        self.__fit_bounds     = [(0, 1000)] * self.lut_entry_values
        self.fitter_state     = FitterState.READY
    
    def fit(self, default = None, xy = None, xyz = None, uv = None):
        if self.fitter_state.value < FitterState.READY.value:
            return False

        # Prepare input values
        rgb = np.zeros((3,))
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
            rgb  = np.array(xyz)
            temp = OptimizerBase.xyzToXy(np.array(xyz))
            if self.is_uv:
                pos = OptimizerBase.xyToUv(temp)
            else:
                pos = temp
        else:
            print('Invalid value call')
            return ([0] * self.lut_entry_values, True, 'Invalid input!')

        if not txy is None:
            rgb[0] = txy[0] * self.rgbsum
            rgb[1] = txy[1] * self.rgbsum
            rgb[2] = (1 - txy[0] - txy[1]) * self.rgbsum
        
        # prepare fitter

        def constrHelper(x):
            xyz = spectra_integrate(np.array(x)[None, ...], self.response[:, 1:])
            return xyz.flatten() - rgb

        x0        = self.__fit_x0.copy()
        bounds    = self.__fit_bounds.copy()
        objective = lambda s: sum([(s[i] - s[i+1])**2 for i in range(len(s)-1)])
        cnstr     = {
                'type': 'eq',
                'fun': constrHelper
            }

        res = minimize(objective, x0, method='SLSQP', constraints=cnstr,
                       bounds=bounds, options={"maxiter": self.maxiter, "ftol": self.ftol})
        
        if not res.success:
            err_message = 'Error for xyz={} after {} iterations: {}'.format(rgb, res.nit, res.message)
            return ([0] * num_bins, True, err_message)
        else:
            # The result may contain some very tiny negative values due
            # to numerical issues. Clamp those to 0.
            return ([max(x, 0) for x in res.x], False, "")
    
    def lookup(self, **kwargs):
        pq, s  = self.toCoordinates(**kwargs)
        lookup = self.bilinearInterpolation(pq)
        if s is None:
            return lookup
        return np.multiply(lookup, s[...,None])

    def saturate(self, data, saturation):
        mval = np.max(data, axis=-1)
        mval_nonzero = mval.copy()
        mval_nonzero[mval_nonzero < self.spectra_epsilon] = 1.0
        return np.multiply(np.power(np.divide(data, mval_nonzero[...,None]), saturation), mval[...,None])

    def toXyz(self, data):
        return spectra_integrate(data, self.xyz[:,1:])