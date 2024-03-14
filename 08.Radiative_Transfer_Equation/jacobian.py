import optical_depth


def calculateJacobian(freq_bounds, atmosphere, abs_species, fnum, upwelling=False, sfc_t, sfc_e):
    profiles, optical_depths, freq = optical_depth.atmo_optical_depth(fbounds=freq_bounds, 
                                                                  atmosphere=atmosphere, 
                                                                  abs_species=optical_depth.getInfraredAbsorbers(),
                                                                  fnum=1_000)

    temp_pert_profiles = profiles.rename({'Temperature':'t', "Pressure":'p', "Height":'z'}, axis=1)
    temp_pert_profiles['t'] += 0.01
    
    profiles_tpert, pert_t_od, freq = optical_depth.atmo_optical_depth(fbounds=freq_bounds, 
                                                                      atmosphere=temp_pert_profiles, 
                                                                      abs_species=optical_depth.getInfraredAbsorbers(),
                                                                      fnum=1_000)

    q_pert_profiles = profiles.rename({'Temperature':'t', "Pressure":'p', "Height":'z'}, axis=1)
    q_pert_profiles['H2O'] =  0.000001 + q_pert_profiles['H2O']
    profiles_qpert, pert_q_od, freq = optical_depth.atmo_optical_depth(fbounds=freq_bounds, 
                                                                      atmosphere=q_pert_profiles, 
                                                                      abs_species=optical_depth.getInfraredAbsorbers(),
                                                                      fnum=1_000)
    zenith = 0
    surface_temperature = 310 # K
    surface_emissivity = 1
    
    wnum = freq.to("1/centimeter", "spectroscopy")
    gas_opd = optical_depths.sum(axis=0).T
    gas_tpert_opd = optical_depths.sum(axis=0).T
    gas_qpert_opd = pert_q_od.sum(axis=0).T
    
    
    temp_profile = profiles['Temperature'].to_numpy()
    ptemp_profile = profiles_tpert['Temperature'].to_numpy()
    rad_dn = rtcalc_module.rt(opd = gas_opd, wnum = wnum, temp = temp_profile, zenith_angle=zenith, upwelling=False,
                              sfc_t=310, sfc_e=1)
    
    jacobian_dn_t = []
    for i in range(len(profiles['Temperature'].to_numpy())):
        pert_temp_profile = temp_profile.copy()
        pert_temp_profile[i] = ptemp_profile[i]
        rad_dn_t = rtcalc_module.rt(opd = gas_opd, wnum = wnum, temp = pert_temp_profile, zenith_angle=zenith, upwelling=False,
                              sfc_t=310, sfc_e=1)
        jacobian_dn_t.append((rad_dn_t - rad_dn)/0.01)
    
    jacobian_dn_q = []
    for i in range(len(profiles['Temperature'].to_numpy())-1):
        pert_q_ods = gas_opd.copy()
        pert_q_ods[i] = gas_qpert_opd[i]
        rad_dn_q = rtcalc_module.rt(opd = pert_q_ods, wnum = wnum, temp = temp_profile, zenith_angle=zenith, upwelling=False,
                              sfc_t=310, sfc_e=1)
        jacobian_dn_q.append((rad_dn_q - rad_dn)/0.000001)

    zenith = 0
    surface_temperature = 310 # K
    surface_emissivity = 1
    
    wnum = freq.to("1/centimeter", "spectroscopy")
    gas_opd = optical_depths.sum(axis=0).T
    gas_tpert_opd = optical_depths.sum(axis=0).T
    gas_qpert_opd = pert_q_od.sum(axis=0).T
    
    
    temp_profile = profiles['Temperature'].to_numpy()
    ptemp_profile = profiles_tpert['Temperature'].to_numpy()
    rad_dn = rtcalc_module.rt(opd = gas_opd, wnum = wnum, temp = temp_profile, zenith_angle=zenith, upwelling=True,
                              sfc_t=temp_profile[0], sfc_e=1)
    
    jacobian_dn_t = []
    for i in range(len(profiles['Temperature'].to_numpy())):
        pert_temp_profile = temp_profile.copy()
        pert_temp_profile[i] = ptemp_profile[i]
        rad_dn_t = rtcalc_module.rt(opd = gas_opd, wnum = wnum, temp = pert_temp_profile, zenith_angle=zenith, upwelling=True,
                              sfc_t=temp_profile[0], sfc_e=1)
        jacobian_dn_t.append((rad_dn_t - rad_dn)/0.01)
    
    jacobian_dn_q = []
    for i in range(len(profiles['Temperature'].to_numpy())-1):
        pert_q_ods = gas_opd.copy()
        #print(gas_qpert_opd.shape, pert_q_ods.shape)
        pert_q_ods[i] = gas_qpert_opd[i]
        rad_dn_q = rtcalc_module.rt(opd = pert_q_ods, wnum = wnum, temp = temp_profile, zenith_angle=zenith, upwelling=True,
                              sfc_t=temp_profile[0], sfc_e=1)
        jacobian_dn_q.append((rad_dn_q - rad_dn)/0.000001)