import numpy as np
import proplot as pplt

def planck(wnum, temp):
    c1 = 1.1910427e-5 #mW/m2.sr.cm-1
    c2 = 1.4387752 #K/cm
    r = (c1 * np.power(wnum,3)) / ( np.exp( c2*wnum/temp) - 1.)
    return r
    
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

def plot_rtm_problem(temperatures, optical_depth, cloud_fraction, cloud_optical_depth, sfc_t, sfc_emissivity, wnum=800, upwelling=False):
    
    fig, ax = pplt.subplots(ncols=1, nrows=1, figsize=(7,4), sharex=False, sharey=False)
    ax.format(ylim=(temperatures.index.min(), temperatures.index.max()), xlim=(0,1),
               ylabel="Height [km]", xticks=[0,0.25,0.5,0.75,1], grid=True)
    
    # Draw the temperatures and Planck function results at the layer boundaries.
    ax.yaxis.set_ticks_position('both')
    ax2 = ax.twinx()
    thermo_ticks = []
    for i, val in enumerate(zip(temperatures, bb_rad)):
        text = rf"              $T_{i}$={val[0]} K, $B_\nu(T)=${val[1]} RU"
        thermo_ticks.append(text)

    # Format the figure to include the ticks indicating the temperatures at layer boundaries
    ax2.format(yticks=np.asarray(temperatures.index), yticklabelcolor='red',
               ylim=(temperatures.index.min(), temperatures.index.max()), yticklabels=thermo_ticks)
    # Format the figure so we know the x-axis denotes cloud fraction.    
    ax.format(ylim=(temperatures.index.min(), temperatures.index.max()), xlim=(0,1),
               ylabel="Height [km]", xlabel="Cloud Fraction ($F$)")

    # Draw the cloud fraction coverage on the plot for layers that have a cloud
    for i, cf in cloud_fraction.items():
        ax.fill_between([0,cf], i-1, i+1, color='b', alpha=0.2)
        
    #ax.yaxis.label.set_color('white')
    
    # Draw the surface properties on the figure
    ax.text(0.77,0.05, rf"$T_{{sfc}}=${sfc_t} K, $B_\nu(T_{{sfc}})=${sfc_bb_rad} RU, $\epsilon_{{sfc}}={sfc_emissivity}$", 
            transform='figure', fontsize=12, horizontalalignment='center', verticalalignment='center')
    
    #ax.text(0.60,0.02, rf"$T_{{sfc}}=${sfc_t}, $B(T_{{sfc}})=${sfc_bb_rad} RU, $\epsilon={sfc_emissivity}$", transform='figure')
    # Draw the gas optical depth values on the plot for each layer.
    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for i, val in optical_depth.items():
        ax.text(1.05,i, rf"$\tau_{{gas}}={val}$", transform=trans, color='#0096FF',
                fontweight='bold', fontsize=12, horizontalalignment='left', verticalalignment='center')
        
    # Draw the cloud optical depth values on the plot for each layer.    
    for i, val in cloud_optical_depth.items():
        print(val)
        if val == 0:
            continue
        ax.text(1.05,i, rf"                        $\tau_{{cloud}}={val}$", transform=trans, color='#808000',
                fontweight='bold', fontsize=12, horizontalalignment='left', verticalalignment='center')

    # Draw the wavenumber the calculation is being done at on the figure.
    ax.text(0.025,.02, rf"$\nu={wnum}\ cm^{{-1}}$", transform='figure', color='green',
                fontweight='bold', fontsize=10)

    # Draw the arrows indicating the layer to instrument transmission (upwelling or downwelling).
    if upwelling is False:
        ax.format(urtitle='Downwelling')
        plt.arrow(0.1, 9, dy=-9, dx=0, width=0.02, head_length=0.5, length_includes_head=True)
        plt.arrow(0.3, 7, dy=-7, dx=0, width=0.02, head_length=0.5, length_includes_head=True)
        plt.arrow(0.5, 5, dy=-5, dx=0, width=0.02, head_length=0.5, length_includes_head=True)
        plt.arrow(0.7, 3, dy=-3, dx=0, width=0.02, head_length=0.5, length_includes_head=True)
        plt.arrow(0.9, 1, dy=-1, dx=0, width=0.02, head_length=0.5, length_includes_head=True)
    elif upwelling is True:
        ax.format(lltitle='Upwelling')
        plt.arrow(0.1, 9, dy=1, dx=0, width=0.02, head_length=0.5, length_includes_head=True)
        plt.arrow(0.3, 7, dy=3, dx=0, width=0.02, head_length=0.5, length_includes_head=True)
        plt.arrow(0.5, 5, dy=5, dx=0, width=0.02, head_length=0.5, length_includes_head=True)
        plt.arrow(0.7, 3, dy=7, dx=0, width=0.02, head_length=0.5, length_includes_head=True)
        plt.arrow(0.9, 1, dy=9, dx=0, width=0.02, head_length=0.5, length_includes_head=True)
