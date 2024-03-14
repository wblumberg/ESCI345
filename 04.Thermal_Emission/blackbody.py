import pint
import numpy as np

ureg = pint.get_application_registry()

c = 2.99792458e8 * ureg.Unit("m/s")# m/s
k_b = 1.380649e-23 * ureg.Unit("J/K") # J K-1
h = 6.62607015e-34 * ureg.Unit("J s") # J s

def rayleigh_jeans(wavelength, temperature):
    wavelength = wavelength.to('m', 'spectroscopy')
    # UV catastrophy
    intensity = (2. * c * k_b * temperature)/np.power(wavelength,4)
    return intensity

def wien_law(wavelength, temperature):
    wavelength = wavelength.to('m', 'spectroscopy')
    multi = (2. * h * c**2)/np.power(wavelength,5.)
    exp = np.exp( - (h * c)/(wavelength * k_b * temperature))
    return multi * exp

def planck(wavelength, temperature):
    wavelength = wavelength.to('m', 'spectroscopy')
    multi = (2. * h * c**2)/np.power(wavelength,5.)
    a = np.exp((h * c)/(wavelength * k_b * temperature))
    intensity = multi * (1./(a - 1.))

    return intensity