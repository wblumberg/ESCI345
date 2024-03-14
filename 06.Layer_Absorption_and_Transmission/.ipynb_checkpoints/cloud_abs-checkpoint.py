import pint
import numpy as np
ureg = pint.get_application_registry()

cloud_thickness = 300 * ureg.meter

from pyrtlib.absorption_model import LiqAbsModel
from pyrtlib.utils import constants

def cloudy_absorption(t, denl, deni, frq):
    """Multiplies cloud density profiles by a given fraction and computes the
    corresponding cloud liquid and ice absorption profiles, using Rosenkranz's
    cloud liquid absorption and ice absorption by [Westwater-1972]_.

    Args:
        t (numpy.ndarray): Temperature profile (k).
        denl (numpy.ndarray): Liquid density profile (:math:`g/m^3`).
        deni (numpy.ndarray): Ice density profile (:math:`g/m^3`).
        frq (numpy.ndarray): Frequency array (GHz).

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: 
        * aliq: Liquid absorption profile (np/km)
        * aice: Ice absorption profile (np/km)

    See also:
        :py:func:`~pyrtlib.absorption_model.LiqAbsModel.liquid_water_absorption`

    """

    nl = len(t)
    c = np.dot(constants('light')[0], 100)
    #print(c)
    ghz2hz = 1e9
    db2np = np.dot(np.log(10.0), 0.1)

    wave = c / (np.dot(frq, ghz2hz))
    LiqAbsModel.model = 'R19SD'
    
    aliq = np.zeros(denl.shape)
    aice = np.zeros(denl.shape)
    for i in range(0, nl):
        # Compute liquid absorption np/km.
        if denl[i] > 0:
            aliq[i] = LiqAbsModel.liquid_water_absorption(
                denl[i], frq, t[i])
        # compute ice absorption (db/km); convert non-zero value to np/km.
        if deni[i] > 0:
            aice[i] = np.dot(
                np.dot((8.18645 / wave), deni[i]), 0.000959553)
            aice[i] = np.dot(aice[i], db2np)

    return aliq, aice

def mw_cloud_model(temperature, LWP, IWP, frq):
    liqabs = []
    iceabs = []
    print(np.array([temperature]), np.array([LWP/(cloud_thickness.magnitude)]), np.array([IWP/(cloud_thickness.magnitude)]))
    for f in frq:
        aliq, aice = cloudy_absorption(np.array([temperature]), np.array([LWP/(cloud_thickness.magnitude)]), np.array([IWP/(cloud_thickness.magnitude)]), f)
        liqabs.append(aliq[0] * cloud_thickness.to('km').magnitude)
        iceabs.append(aice[0] * cloud_thickness.to('km').magnitude)
    return liqabs, iceabs


def scattering_regimes(x):
    if x < 0.002:
        print("Scattering Regime: Negligible Scattering")
    elif x < 0.2 and x >= 0.002:
        print("Scattering Regime: Rayleigh")
    elif x < 2000 and x >= 0.2:
        print("Scattering Regime: Mie")
    else:
        print("Scattering Regime: Geometric Optics")

# Our approximate equation for the cloud optical depth tau* is
def vis_cloud_od(lwp, reff):
    # Assumes Qe = 2 because this is true when the particles are large compare to the wavelength
    # of radiation
    rho_l = 1000 * ureg.kilogram / (ureg.meter ** 3) # kg/m3 (density of pure water)
    return (3./2.) * lwp.to_base_units() / (rho_l.to_base_units() * reff.to_base_units())

def od2numdens(ext_coef, thickness, od):
    extcoefxsec = ext_coef.pint.dequantify().values * ureg.Unit("meter**2")
    extcoefvol = od / thickness
    return extcoefvol / extcoefxsec

def numdens2od(ext_coef, thickness, numdens):
    extcoefxsec = ext_coef.pint.dequantify().values * ureg.Unit("meter**2")
    extcoefvol = extcoefxsec * numdens.values
    return (extcoefvol * thickness).magnitude

def get_ir_cloud_ods(ssp, intau, wnum, reff, thickness=cloud_thickness):
    wnum_reff = 900 * ureg.Unit('1/centimeter')

    ref_to_wlen = wnum_reff.to('micrometer', 'spectroscopy')
    wnum_to_wlen = wnum.to('micrometer', 'spectroscopy')
    reff = reff.to('micrometer')

    Qext = ssp['Qext'].interp(wavelength=ref_to_wlen, reff=reff)
    kext = ssp['ext'].interp(wavelength=ref_to_wlen, reff=reff)

    numdens = od2numdens(kext, thickness.to('meter'), intau*Qext/2.)

    kabs = ssp['abs'].interp(wavelength=wnum_to_wlen.magnitude, reff=reff.magnitude)
    tauout = numdens2od(kabs, thickness, numdens)
    return tauout