import optical_depth
import rtcalc_module
import numpy as np

def calculateTempJacobian(freq_bounds, atmosphere, abs_species, zenith=0, fnum=1_000, 
                       upwelling=True, sfc_t=310, sfc_e=1, deltaT=0.01):
    ## CALCULATES A TEMPERATURE JACOBIAN
    
    ### Step 1: Compute the gas optical depths for our different scenarios.
    # Baseline calculation of the gas optical depths
    profiles, optical_depths, freq = optical_depth.atmo_optical_depth(fbounds=freq_bounds, 
                                                                  atmosphere=atmosphere, 
                                                                  abs_species=abs_species,
                                                                  fnum=fnum)

    # Perturbed calculation of the gas optical depths (temperature pert.)
    temp_pert_profiles = profiles.rename({'Temperature':'t', "Pressure":'p', "Height":'z'}, axis=1) # rename the DataFrame
    temp_pert_profiles['t'] += deltaT # Perturb by 0.01 K
    
    profiles_tpert, pert_t_od, freq = optical_depth.atmo_optical_depth(fbounds=freq_bounds, 
                                                                      atmosphere=temp_pert_profiles, 
                                                                      abs_species=abs_species,
                                                                      fnum=fnum)
    
    
    ## Convert the spectral grid back to wavenumber space and calculate the layer-optical depths
    wnum = freq.to("1/centimeter", "spectroscopy")
    gas_opd = optical_depths.sum(axis=0).T
    gas_tpert_opd = optical_depths.sum(axis=0).T
    
    
    temp_profile = profiles['Temperature'].to_numpy() # Baseline temperature profile
    ptemp_profile = profiles_tpert['Temperature'].to_numpy() # Perturbed temperature profile
    
    # Compute the baseline radiances
    # 1.) Downwelling radiance
    rad_dn = rtcalc_module.rt(opd = gas_opd, wnum = wnum, temp = temp_profile, zenith_angle=zenith, upwelling=upwelling,
                              sfc_t=sfc_t, sfc_e=sfc_e) 

    
    # 2.) Upwelling radiance
    rad_up = rtcalc_module.rt(opd = gas_opd, wnum = wnum, temp = temp_profile, zenith_angle=zenith, upwelling=True,
                              sfc_t=sfc_t, sfc_e=sfc_e) 
    
    ### Calculate the Jacobians for ground-based and space-based observations
    jacobian_dn_t = []
    jacobian_up_t = []

    for i in range(len(profiles['Temperature'].to_numpy())):
        pert_temp_profile = temp_profile.copy()
        pert_temp_profile[i] = ptemp_profile[i]
        
        rad_dn_t = rtcalc_module.rt(opd = gas_opd, wnum = wnum, temp = pert_temp_profile, zenith_angle=zenith, upwelling=False,
                              sfc_t=sfc_t, sfc_e=sfc_e) # Compute perturbed temperature downwelling calc.
        rad_up_t = rtcalc_module.rt(opd = gas_opd, wnum = wnum, temp = pert_temp_profile, zenith_angle=zenith, upwelling=True,
                              sfc_t=sfc_t, sfc_e=sfc_e) # Compute perturbed temperature upwelling calc.
        
        jacobian_dn_t.append((rad_dn_t - rad_dn)/deltaT) # Compute the Jacobian (downwelling)
        jacobian_up_t.append((rad_up_t - rad_up)/deltaT) # Compute the Jacobian (upwelling)
    
    return wnum, np.asarray(jacobian_dn_t), np.asarray(jacobian_up_t)
        
        
        
def calculateGasJacobian(freq_bounds, atmosphere, abs_species, gas_species='H2O', zenith=0, fnum=1_000, 
                       upwelling=True, sfc_t=310, sfc_e=1, deltaQ=0.000001):        
        
    ## CALCULATES A GAS JACOBIAN
    
    ### Step 1: Compute the gas optical depths for our different scenarios.
    # Baseline calculation of the gas optical depths
    profiles, optical_depths, freq = optical_depth.atmo_optical_depth(fbounds=freq_bounds, 
                                                                  atmosphere=atmosphere, 
                                                                  abs_species=abs_species,
                                                                  fnum=fnum)
    
    # Perturbed calculation of the gas optical depths (trace gas pert.)
    q_pert_profiles = profiles.rename({'Temperature':'t', "Pressure":'p', "Height":'z'}, axis=1) # rename the DataFrame
    q_pert_profiles[gas_species] = deltaQ + q_pert_profiles[gas_species] # perturb by 0.000001
    profiles_qpert, pert_q_od, freq = optical_depth.atmo_optical_depth(fbounds=freq_bounds, 
                                                                      atmosphere=q_pert_profiles, 
                                                                      abs_species=abs_species,
                                                                      fnum=fnum)

    ### Step 2: Compute the Jacobians by swapping out each layer and computing the radiances
    
    ## Convert the spectral grid back to wavenumber space and calculate the layer-optical depths
    wnum = freq.to("1/centimeter", "spectroscopy")
    gas_opd = optical_depths.sum(axis=0).T
    gas_qpert_opd = pert_q_od.sum(axis=0).T
    
    temp_profile = profiles['Temperature'].to_numpy() # Baseline temperature profile
    
    # Compute baseline downwelling radiance
    rad_dn = rtcalc_module.rt(opd = gas_opd, wnum = wnum, temp = temp_profile, zenith_angle=zenith, upwelling=False,
                              sfc_t=sfc_t, sfc_e=sfc_e) 
    
    # Compute baseline upwelling radiance
    rad_up = rtcalc_module.rt(opd = gas_opd, wnum = wnum, temp = temp_profile, zenith_angle=zenith, upwelling=True,
                              sfc_t=sfc_t, sfc_e=sfc_e) 
    
    jacobian_dn_q = []
    jacobian_up_q = []
    for i in range(len(profiles['Temperature'].to_numpy())-1):
        pert_q_ods = gas_opd.copy()
        pert_q_ods[i] = gas_qpert_opd[i]
        
        rad_dn_q = rtcalc_module.rt(opd = pert_q_ods, wnum = wnum, temp = temp_profile, zenith_angle=zenith, upwelling=False,
                              sfc_t=sfc_t, sfc_e=sfc_e)
        rad_up_q = rtcalc_module.rt(opd = pert_q_ods, wnum = wnum, temp = temp_profile, zenith_angle=zenith, upwelling=True,
                              sfc_t=sfc_t, sfc_e=sfc_e) # Compute perturbed gas upwelling calc
        
        jacobian_up_q.append((rad_up_q - rad_up)/deltaQ) # Compute the Jacobian (upwelling)
        jacobian_dn_q.append((rad_dn_q - rad_dn)/deltaQ) # Compute the Jacobian (downwelling)
    
    return wnum, np.asarray(jacobian_dn_q), np.asarray(jacobian_up_q)