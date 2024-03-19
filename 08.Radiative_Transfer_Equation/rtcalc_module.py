import numpy as np
import pint

ureg = pint.get_application_registry()

def planck(wnum, temp):
    c1 = 1.1910427e-5 #mW/m2.sr.cm-1
    c2 = 1.4387752 #K/cm
    r = (c1 * np.power(wnum,3)) / ( np.exp( c2*wnum/temp) - 1.)
    return r * ureg.Unit("mW.(m^-2.sr^-1.cm^-1)") #mW/m2.sr.cm-1

def planck_wavelength(wlen, temp):
    wlen = wlen.to('micrometer', 'spectroscopy')
    c1 = 1.1911e8 #mW/m2.sr.µm
    c2 = 1.439e4 #K/cm
    r = (1000 * c1 * np.power(wlen, -5)) / ( np.exp( c2/(wlen*temp) - 1.)) 
    return r * ureg.Unit("mW.(m^-2.sr^-1.micrometer)") #mW/m2.sr.cm-1

def brightness_temperature(rad, wnum):
    #wnum = wnum.to('1/cm', 'spectroscopy')
    c1 = 1.1910427e-5 #mW/m2.sr.cm-1
    c2 = 1.4387752 #K/cm
    temp = (c2 * wnum) / np.log( ((c1 * np.power(wnum,3))/rad) + 1) 
    return temp * ureg.Unit("K") # Kelvin 

def rt(wnum, temp, opd, zenith_angle=0, sfc_t=None, sfc_e=None, upwelling=False, debug=False):
    opd = np.asarray(opd, dtype=np.float64) * (1./np.cos(np.radians(zenith_angle)))
    wnum = np.asarray(wnum, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    if upwelling is False:
        sfce = 0.
        k_start = len(opd)-1
        k_end = -1
        k_step = -1
    else:
        sfce = sfc_e
        if sfce < 0 or sfce > 1:
            print("Error: the surface emissivity is outside of [0,1]")
            sys.exit()        
        if sfc_t is None:
            # if no surface temperature is given, use the temperature of the lowest level
            sfct = temp[0]
        else:
            sfct = sfc_t
        sfc_b = planck(wnum, sfct)
        k_start = 0
        k_end = len(opd)
        k_step = 1

    rad = np.zeros(len(wnum))

    # use the linear in tau approach
    if sfce > 0:
        rad = sfc_b * sfce + rt(wnum, temp, opd) * (1.-sfce)
    # B_close is the Planck function for the temperature at the edge of the layer 
    # B_far | --> | B_close
    # 
    for k in np.arange(k_start, k_end, k_step):
        trans = np.asarray(np.exp(-1. * opd[k,:]), dtype=np.float64) # Compute the transmissivity of the layer
        if upwelling is True:
            # Compute the b_boundary from the bottom layer up
            b_close = planck(wnum, temp[k+1])
            layer_to_inst = np.exp(-1. * np.sum(opd[:k,:], axis=0))
        else:
            # Compute the b_boundary from the top of the layer down
            b_close = planck(wnum, temp[k])
            layer_to_inst = np.exp(-1. * np.sum(opd[k+1:,:], axis=0))
        b_avg = (planck(wnum, temp[k]) + planck(wnum, temp[k+1]))/2. # b_close and b_far
        b_eff = b_close + 2*(b_avg - b_close)*((1./opd[k,:]) - (trans/(1.-trans)))
        if np.isnan(b_eff).all():
            b_eff = b_avg
        if debug == True:
            print("Temperature of top and bottom layer:", temp[k], temp[k+1])
            print("Planck top and bottom layer:", planck(wnum, temp[k]), planck(wnum, temp[k+1]))
            print("b_avg:", b_avg)
            print("Temperature of top and bottom layer:", temp[k], temp[k+1])
            print("b_close:", b_close) 
            print("b_avg:", b_avg)
            print("b_eff:", b_eff)
            print("Optical Depth of Current Layer:", opd[k,:])
            print("Terms of the RT for this layer:", (1-np.exp(-1.*opd[k,:])), b_avg, layer_to_inst)
            print("Calculation:", (1-np.exp(-1.*opd[k,:]))*b_eff*layer_to_inst) 
        rad = rad*trans + (1.-trans) * b_eff
    return rad