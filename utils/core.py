import numpy as np
import astropy.units as u
from scipy.interpolate import interp1d

class Snell(object):
    """Implementation of Snell's law of reflection.

    The initialization takes two refractive indexes, and the default for
    the refractive index of the second medium is 1.

    >>> s = Snell(1.5)
    >>> print(s.n1)
    1.5
    >>> print(s.n2)
    1.0

    The methods `.angle1` and `.angle2` calculate the angle in medium 1 and
    2, respectively, from the angle in the other medium.
    """

    def __init__(self, n1, n2=1.):
        """
        n1: number, refractive index of the first media
        n2: optional, number, refractive index of the second media
        """
        self.n1 = n1
        self.n2 = n2

    @staticmethod
    def refraction_angles(angle, n1, n2):
        """
        Calculates the refractive angle in the second media from that in the
        first media.

        angle: number, astropy Quantity, or array-like number or Quantity,
            the angle(s) in the first media.  If not Quantity, then in
            degrees by default.
        n1: number, refractive index of the first (incident) media
        n2: number, refractive index of the second (emission) media
        """
        sinangle = n1/n2 * np.sin(np.deg2rad(angle))
        if sinangle > 1:
            return np.nan
        a = np.arcsin(sinangle)
        if not isinstance(angle, u.Quantity):
            a = np.rad2deg(a)
        return a

    def angle1(self, angle2):
        """
        Calculates the refractive angle in the first media from that in the
        second media.

        angle2: number, astropy Quantity, or array-like number or Quantity,
            the angle(s) in the second media.  If not Quantity, then in
            degrees by default.
        """
        return Snell.refraction_angles(angle2, self.n2, self.n1)

    def angle2(self, angle1):
        """
        Calculates the refractive angle in the second media from that in the
        first media.

        angle2: number, astropy Quantity, or array-like number or Quantity,
            the angle(s) in the second media.  If not Quantity, then in
            degrees by default.
        """
        return Snell.refraction_angles(angle1, self.n1, self.n2)

    @property
    def critical_angle(self):
        """Critial angle of reflection"""
        n1 = self.n1
        n2 = self.n2
        if n1 > n2:
            n1, n2 = n2, n1
        return np.rad2deg(np.arcsin(n1/n2))

    @property
    def brewster_angle(self):
        """Brewster's angle

        Calculated for light transmiting from medium 1 (n1) to medium 2 (n2)
        """
        return np.rad2deg(np.arctan(self.n2/self.n1))

    def reflectance_coefficient(self, angle1=None, angle2=None, pol=None):
        """Calculate reflectance coefficient

        Parameter
        ---------
        angle1 : number, `astropy.units.Quantity`
            Angle in the 1st medium.  Only one of `angle1` or `angle2` should
            be passed.  If both passed, then an error will be thrown.
        angle2 : number, `astropy.units.Quantity`
            Angle in the 2nd medium.  Only one of `angle1` or `angle2` should
            be passed.  If both passed, then an error will be thrown.
        pol : in ['s', 'normal', 'perpendicular', 'p', 'in plane', 'parallel']
            The polarization state for calculation.
            ['s', 'normal', 'perpendicular']: normal to plane of incidence
            ['p', 'in plane', 'parallel']: in the plan of incidence
            Default will calculate the average of both polarization states

        Return
        ------
        Reflection coefficient
        """
        if (angle1 is not None) and (angle2 is not None):
            raise ValueError('ony one angle should be passed')
        if (angle1 is None) and (angle2 is None):
            raise ValueError('one angle has to be passed')
        if angle1 is not None:
            angle2 = self.angle2(angle1)
        if angle2 is not None:
            angle1 = self.angle1(angle2)
        if pol in ['s', 'normal', 'perpendicular', None]:
            a = self.n1 * np.cos(np.deg2rad(angle1))
            b = self.n2 * np.cos(np.deg2rad(angle2))
            Rs = ((a - b)/(a + b))**2
            if not np.isfinite(Rs):
                Rs = 1.0
        if pol in ['p', 'in plane', 'parallel', None]:
            a = self.n1 * np.cos(np.deg2rad(angle2))
            b = self.n2 * np.cos(np.deg2rad(angle1))
            Rp = ((a - b)/(a + b))**2
            if not np.isfinite(Rp):
                Rp = 1.0
        try:
            return (Rs + Rp)/2
        except NameError:
            try:
                return Rs
            except NameError:
                return Rp


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
            
    def emission(self, emi_ang, wavelength, epsrel=1e-4, debug=False):
        """Calculates the quantity at the surface with subsurface emission
        propagated and accounted for.

        emi_ang: number or astropy Quantity, emission angle.  If not Quantity,
            then angle is in degrees
        wavelength: wavelength to calculate, same unit as the length quantities
            in `Surface.layers` class objects
        epsrel: optional, relative error to tolerance in numerical
            integration.  See `scipy.integrate.quad`.
        """
        if hasattr(emi_ang,'__iter__'):
            raise ValueError('`emi_ang` has to be a scaler')
        self._check_layer_depth()
        self._check_layer_profile()

        L = 0    # total path length of light in the unit of absorption length
        n0 = 1.  # adjacent media outside of the layer to be calculated
        m = 0.   # integrated quantity
        trans_coef = 1.  # transmission coefficient = 1 - ref_coef
        if debug:
            D = 0
            prof = {'t': [], 'intprofile': [], 'zzz': [], 'L0': []}
        for i,l in enumerate(self.layers):
            # integrate in layer `l`
            snell = Snell(l.n, n0)
            inc = snell.angle1(emi_ang)
            ref_coef = snell.reflectance_coefficient(angle2=emi_ang)
            coef = l.absorption_coefficient(wavelength)
            cos_i = np.cos(np.deg2rad(inc))
            intfunc = lambda z: l.profile(z) * np.exp(-coef*z/cos_i-L)
            dd = -2.3026*np.log10(epsrel)/coef
            #dd = l.profile.x.max()
            if l.depth > dd:
                zz = np.linspace(0, dd, 1000)
            else:
                zz = np.linspace(0, l.depth, 1000)
            if debug:
                prof['t'].append(l.profile(zz))
                prof['intprofile'].append(intfunc(zz))
                prof['zzz'].append(zz+D)
                prof['L0'].append(L)
                D += l.depth
            # from scipy.integrate import quad
            # integral = quad(intfunc, 0, l.depth, epsrel=epsrel)[0]
            integral = (intfunc(zz)[:-1]*(zz[1:]-zz[:-1])).sum()
            trans_coef *= (1-ref_coef)
            m += trans_coef*coef*integral/cos_i
            # prepare for the next layer
            L += l.depth/cos_i*coef
            emi_ang = inc
            n0 = l.n
            if debug:
                print(f'cos(i) = {cos_i}, coef = {coef}, ref_coef = {ref_coef}, integral = {integral}')

        if debug:
            return m, prof
        else:
            return m