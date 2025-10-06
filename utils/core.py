import numpy as np
import astropy.units as u
from scipy.interpolate import interp1d

class Layer(object):
    """Layer class for calculating propagation of subsurface thermal emission

    """

    def __init__(self, n, loss_tangent, depth=np.inf, profile=None):
        """
        Parameters
        ----------
        n : number
            Real part of refractive index
        loss_tangent : number
            Loss tengent
        depth : optional, number
            Depth of the layer in the same unit as `z` (see below for the
            description of `profile`)
        profile : optional, callable object or array-like of shape (2, N)
            If callable, then `profile(z)` returns the physical quantity of
            the layer at depth `z`.  Most commonly it is a temperature
            profile, or a thermal emission profile.
            If array-like, then profile[0] is depth and profile[1] is the
            physical quantity at corresponding depth.
        """
        self.n = n
        self.depth = depth
        self.loss_tangent = loss_tangent
        if profile is None:
            self.profile = None
        elif hasattr(profile, '__call__'):
            self.profile = profile
        elif np.shape(profile)[0] == 2:

            self.profile = interp1d(profile[0], profile[1], bounds_error=False,
                                    fill_value=(profile[1][0], profile[1][-1]))
        else:
            raise ValueError('Unrecogniazed type or format for `profile`')

    def absorption_length(self, *args, **kwargs):
        """
        Calculates absorption length

        See `.absorption_coefficient`
        """
        return 1./self.absorption_coefficient(*args, **kwargs)

    def absorption_coefficient(self, wave_freq):
        """
        Calculates absorption coefficient.
        Follow Hapke (2012) book, Eq. 2.84, 2.85, 2.95b

        wave_freq : number, `astropy.units.Quantity` or array_like
            Wavelength or frequency of observation.  Default unit is
            'm' if not specified.
        """
        if isinstance(wave_freq, u.Quantity):
            wavelength = wave_freq.to(u.m, equivalencies=u.spectral())
        else:
            wavelength = wave_freq
        c = np.sqrt(1+self.loss_tangent*self.loss_tangent)
        return (4*np.pi*self.n)/wavelength*np.sqrt((c-1)/(c+1))

class Surface(object):
    """
    Surface class that contain subsurface layers and calculates the observables
    from the surface, such as brightness temperature or thermal emission, etc.

    Based on the models in Keihm & Langseth (1975), Icarus 24, 211-230
    """

    def __init__(self, layers, profile=None):
        """
        layers: SubsurfaceLayer class object or array_like of it, the
            subsurface layers.  If array-like, then the layers are ordered
            from the top surface downward.
        profile : optional, array-like of shape (2, N)
            `profile[0]` is depth and `profile[1]` is the physical quantity at
            corresponding depth.
            If this parameter is specified, it will override the `.profile`
            properties of input layers.  This is the mechanism to provide a
            continuous temperature profile for multi-layered model.
        """
        if not hasattr(layers, '__iter__'):
            self.layers = [layers]
        else:
            self.layers = layers
        self.depth = 0
        self._check_layer_depth()
        self.depth = np.sum([l.depth for l in self.layers])
        self.n_layers = len(self.layers)
        self.profile = profile
        # set `.profile` properties for all layers
        if profile is not None:
            if hasattr(profile, '__call__'):
                for l in self.layers:
                    l.profile = profile
            elif np.shape(profile)[0] == 2:
                prof_int = interp1d(profile[0], profile[1], bounds_error=False,
                                    fill_value=(profile[1][0], profile[1][-1]))
                if self.n_layers == 1:
                    self.layers[0].profile = prof_int
                else:
                    z0 = 0
                    for l in self.layers:
                        l.profile = interp1d(profile[0]-z0, profile[1],
                                    bounds_error=False,
                                    fill_value=(profile[1][0], profile[1][-1]))
                        z0 += l.depth
            else:
                raise ValueError('Unrecogniazed type or format for `profile`')

    def _check_layer_depth(self):
        for i,l in enumerate(self.layers[:-1]):
            if l.depth == np.inf:
                raise ValueError('only the deepest layer can have infinite '
                    'depth.  The depth of the {}th layer cannot be '
                    'infinity'.format(i))


    def _check_layer_profile(self):
        for i,l in enumerate(self.layers):
            if l.profile is None:
                raise ValueError('the {}th layer does not have a quantity '
                    'profile defined'.format(i))
            if not hasattr(l.profile, '__call__'):
                raise ValueError('the {}th layer does not have a valid '
                    'quantity profile defined'.format(i))