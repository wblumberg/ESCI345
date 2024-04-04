"""Simulate and plot Earth's outgoing longwave radiation (OLR)."""
import numpy as np
import pyarts.workspace
from typhon import physics as phys
import matplotlib.pyplot as plt
import proplot as pplt
import climlab
from climlab.domain import field

def calc_spectral_irradiance(atmfield,
                             nstreams=2,
                             fnum=300,
                             fmin=1.0,
                             fmax=97e12,
                             verbosity=0):
    """Calculate the spectral downward and upward irradiance for a given atmosphere.
    Irradiandce is defined as positive quantity independent of direction.

    Parameters:
        atmfield (GriddedField4): Atmosphere field.
        nstreams (int): Even number of streams to integrate the radiative fluxes.
        fnum (int): Number of points in frequency grid.
        fmin (float): Lower frequency limit [Hz].
        fmax (float): Upper frequency limit [Hz].
        verbosity (int): Reporting levels between 0 (only error messages)
            and 3 (everything).

    Returns:
        ndarray, ndarray, ndarray, ndarray, ndarray, ndarray :
        Frequency grid [Hz], altitude [m], pressure [Pa], temperature [K],
        spectral downward irradiance [Wm^-2 Hz^-1],
        spectral upward irradiance [Wm^-2 Hz^-1].
    """
    ws = pyarts.workspace.Workspace(verbosity=0)
    ws.water_p_eq_agendaSet()
    ws.gas_scattering_agendaSet()
    ws.PlanetSet(option="Earth")
    ws.verbositySetScreen(ws.verbosity, verbosity)

    # (standard) emission calculation
    ws.iy_main_agendaSet(option="Emission")

    # cosmic background radiation
    ws.iy_space_agendaSet(option="CosmicBackground")

    # standard surface agenda (i.e., make use of surface_rtprop_agenda)
    ws.iy_surface_agendaSet(option="UseSurfaceRtprop")

    # sensor-only path
    ws.ppath_agendaSet(option="FollowSensorLosPath")

    # no refraction
    ws.ppath_step_agendaSet(option="GeometricPath")

    # Number of Stokes components to be computed
    ws.IndexSet(ws.stokes_dim, 1)

    # No jacobian calculation
    ws.jacobianOff()

    # Clearsky = No scattering
    ws.cloudboxOff()

    # Definition of species
    #ws.abs_speciesSet(species=[
    #    "H2O, H2O-SelfContCKDMT400, H2O-ForeignContCKDMT400",
    #    "CO2, CO2-CKDMT252",
    #    "O3-*", "N2O-*", "CH4-*", "CO-*"
    #])

    ws.abs_speciesSet(species=[
        "H2O, H2O-SelfContCKDMT400, H2O-ForeignContCKDMT400",
        "CO2, CO2-CKDMT252",
        "O3-*", "CH4-*", "O2-*", "N2O-*"
    ])
    
    # Read line catalog
    ws.abs_lines_per_speciesReadSpeciesSplitCatalog(basename="lines/")

    # Load CKDMT400 model data
    ws.ReadXML(ws.predefined_model_data, "model/mt_ckd_4.0/H2O.xml")

    # ws.abs_lines_per_speciesLineShapeType(option=lineshape)
    ws.abs_lines_per_speciesCutoff(option="ByLine", value=750e9)
    # ws.abs_lines_per_speciesNormalization(option=normalization)

    # Create a frequency grid
    ws.VectorNLinSpace(ws.f_grid, int(fnum), float(fmin), float(fmax))

    # Throw away lines outside f_grid
    ws.abs_lines_per_speciesCompact()

    # Weakly reflecting surface
    ws.VectorSetConstant(ws.surface_scalar_reflectivity, 1, 0.0)
    ws.surface_rtprop_agendaSet(
        option="Specular_NoPol_ReflFix_SurfTFromt_surface")

    # No sensor properties
    ws.sensorOff()

    # Atmosphere and surface
    ws.atm_fields_compact = atmfield
    ws.AtmosphereSet1D()
    ws.AtmFieldsAndParticleBulkPropFieldFromCompact()

    # Set surface height and temperature equal to the lowest atmosphere level
    ws.Extract(ws.z_surface, ws.z_field, 0)
    ws.Extract(ws.t_surface, ws.t_field, 0)

    """
    The possible choices for iy_unit are
     "1"             : No conversion, i.e. [W/(m^2 Hz sr)] (radiance per
                         frequency unit).
     "RJBT"          : Conversion to Rayleigh-Jean brightness
                         temperature.
     "PlanckBT"      : Conversion to Planck brightness temperature.
     "W/(m^2 m sr)"  : Conversion to [W/(m^2 m sr)] (radiance per
                         wavelength unit).
     "W/(m^2 m-1 sr)": Conversion to [W/(m^2 m-1 sr)] (radiance per
                         wavenumber unit).
    """
    
    # Output radiance not converted
    #ws.StringSet(ws.iy_unit, "W/(m^2 m-1 sr)") - this doesn't work, the spectral irradiance is always output in W/(m2 Hz)
    ws.StringSet(ws.iy_unit, "1")

    # Definition of sensor position and LOS
    ws.MatrixSet(ws.sensor_pos, np.array([[100e3]]))  # sensor in z = 100 km
    ws.MatrixSet(ws.sensor_los,
                 np.array([[180]
                           ]))  # zenith angle: 0 looking up, 180 looking down

    # Perform RT calculations
    ws.propmat_clearsky_agendaAuto()
    ws.lbl_checkedCalc()
    ws.atmfields_checkedCalc()
    ws.atmgeom_checkedCalc()
    ws.cloudbox_checkedCalc()
    ws.sensor_checkedCalc()

    ws.AngularGridsSetFluxCalc(N_za_grid=nstreams,
                               N_aa_grid=1,
                               za_grid_type="double_gauss")

    # calculate intensity field
    ws.Tensor3Create("trans_field")
    ws.spectral_radiance_fieldClearskyPlaneParallel(trans_field=ws.trans_field,
                                                    use_parallel_za=0)
    ws.spectral_irradiance_fieldFromSpectralRadianceField()

    spectral_flux_downward = -ws.spectral_irradiance_field.value[:, :, 0, 0,
                                                                 0].copy()
    spectral_flux_upward = ws.spectral_irradiance_field.value[:, :, 0, 0,
                                                              1].copy()

    spectral_flux_downward[np.isnan(spectral_flux_downward)] = 0.
    spectral_flux_upward[np.isnan(spectral_flux_upward)] = 0.

    # set outputs
    f = ws.f_grid.value[:].copy()
    z = ws.z_field.value[:].copy().squeeze()
    p = atmfield.grids[1][:].squeeze().copy()
    T = atmfield.get("T")[:].squeeze().copy()

    return f, z, p, T, spectral_flux_downward, spectral_flux_upward


def calc_irradiance(atmfield,
                    nstreams=2,
                    fnum=300,
                    fmin=1.0,
                    fmax=97e12,
                    verbosity=0):
    """Calculate the downward and upward irradiance for a given atmosphere.
    Irradiandce is defined as positive quantity independent of direction.

    Parameters:
        atmfield (GriddedField4): Atmosphere field.
        nstreams (int): Even number of streams to integrate the radiative fluxes.
        fnum (int): Number of points in frequency grid.
        fmin (float): Lower frequency limit [Hz].
        fmax (float): Upper frequency limit [Hz].
        verbosity (int): Reporting levels between 0 (only error messages)
            and 3 (everything).

    Returns:
        ndarray, ndarray, ndarray, ndarray, ndarray :
        Altitude [m], pressure [Pa], temperature [K],
        downward irradiance [Wm^-2], upward irradiance [Wm^-2].
    """

    f, z, p, T, spectral_flux_downward, spectral_flux_upward = calc_spectral_irradiance(
        atmfield,
        nstreams=nstreams,
        fnum=fnum,
        fmin=fmin,
        fmax=fmax,
        verbosity=verbosity)

    #calculate flux
    flux_downward = np.trapz(spectral_flux_downward, f, axis=0)
    flux_upward = np.trapz(spectral_flux_upward, f, axis=0)

    return z, p, T, flux_downward, flux_upward


def integrate_spectral_irradiance(f, spectral_flux, fmin=-np.inf, fmax=np.inf):
    """Calculate the integral of the spectral iradiance from fmin to fmax.

    Parameters:
        f (ndarray): Frequency grid [Hz].
        spectral_flux (ndarray): Spectral irradiance [Wm^-2 Hz^-1].
        fmin (float): Lower frequency limit [Hz].
        fmax (float): Upper frequency limit [Hz].

    Returns:
        ndarray irradiance [Wm^-2].
    """

    logic = np.logical_and(fmin <= f, f < fmax)

    flux = np.trapz(spectral_flux[logic, :], f[logic], axis=0)

    return flux

def atmosphere_plotter(atmosphere):
    fig, axs = pplt.subplots(ncols=3, nrows=2, figsize=(12,8), sharex=False, sharey=False)
    fig.format(ylim=(1000,.1), ylabel="Pressure [mb]")
    
    axs[0].plot(atmosphere['H2O'], atmosphere['p']/100., label="H2O")
    axs[0].plot(atmosphere['O3'], atmosphere['p']/100., label="O3")
    axs[0].plot(atmosphere['CO2'], atmosphere['p']/100., label="CO2")
    axs[0].plot(atmosphere['CH4'], atmosphere['p']/100., label="CH4")
    axs[0].plot(atmosphere['N2O'], atmosphere['p']/100., label="N2O")
    axs[0].plot(atmosphere['O2'], atmosphere['p']/100., label="O2")
    axs[0].format(xscale='log', xlim=(1e-10,1), yscale='log', xformatter='sci', yreverse=True, ylabel="Pressure [mb]", xlabel="Volumetric Mixing Ratio", title="Gaseous Absorbers")
    axs[0].legend()
    
    axs[1].plot(atmosphere['Cloud_LWP'], atmosphere['p']/100., label="LWP")
    axs[1].plot(atmosphere['Cloud_IWP'], atmosphere['p']/100., label="IWP")
    axs[1].format(yscale='log', yreverse=True, ylabel="Pressure [mb]", xlabel="Water Path [$g/m^2$]", title="Ice/Liquid Cloud Water Path", xlim=(0,500))
    axs[1].legend(ncols=1)
    
    axs[2].plot(atmosphere['R_eff_liq'], atmosphere['p']/100., label="$R_{eff,liq}$")
    axs[2].plot(atmosphere['R_eff_ice'], atmosphere['p']/100., label="$R_{eff,ice}$")
    axs[2].legend(ncols=1)
    axs[2].format(yscale='log', yreverse=True, ylabel="Pressure [mb]", xlabel="Effective Radius [µm]", 
                  title="Cloud Effective Radius", xlim=(0,1))
    
    axs[3].plot(np.exp(-atmosphere['SW_Cloud_Tau']), atmosphere['p']/100., label="SW Cloud")
    axs[3].plot(np.exp(-atmosphere['SW_Aerosol_Tau']), atmosphere['p']/100., label="SW Aerosol")
    axs[3].plot(np.exp(-atmosphere['LW_Cloud_Tau']), atmosphere['p']/100., label="LW Cloud")
    axs[3].plot(np.exp(-atmosphere['LW_Aerosol_Tau']), atmosphere['p']/100., label="LW Aerosol")
    axs[3].plot(atmosphere['Cloud_Fraction'], atmosphere['p']/100., label="Cloud Fraction")
    
    axs[3].legend(ncols=1)
    axs[3].format(yscale='log', yreverse=True, ylabel="Pressure [mb]", xlabel="Transmission", 
                  title="Cloud/Aerosol Extinction and Cloud Fraction", xlim=(0,1))
    
    axs[4].plot(atmosphere['SW_Cloud_SSA'], atmosphere['p']/100., label="SW Cloud $\widetilde\omega$")
    axs[4].plot(atmosphere['SW_Cloud_Asym'], atmosphere['p']/100., label="SW Cloud $g$")
    axs[4].plot(atmosphere['SW_Aerosol_SSA'], atmosphere['p']/100., label="SW Aerosol $\widetilde\omega$")
    axs[4].plot(atmosphere['SW_Aerosol_Asym'], atmosphere['p']/100., label="SW Aerosol $g$")
    axs[4].legend(ncols=1)
    axs[4].format(yscale='log', yreverse=True, ylabel="Pressure [mb]", xlabel="Scattering Parameter", 
                  title="Cloud/Aerosol Scattering Properties", xlim=(-1,1))
    
    axs[5].plot(atmosphere['t'], atmosphere['p']/100.)
    axs[5].format(yscale='log', yreverse=True, ylabel="Pressure [mb]", xlabel="Temperature [K]",
                 title="Atmospheric Temperature")

    return fig

def abs_species(df, var):
    data = df[var].to_numpy()#[np.newaxis, ...]
    #lyr_avg = (data[1:] + data[:-1])/2.
    return data#[np.newaxis, ...]


def get_rrtmg_params(rrtmg):
    params = {'insolation': rrtmg.insolation,
              'OLR': rrtmg.OLR[0],
              'ASR': rrtmg.ASR[0],
              'ASRclr': rrtmg.ASRclr[0],
              'ASRcld': rrtmg.ASRcld[0],
              'OLRclr': rrtmg.OLRclr[0],
              'OLRcld': rrtmg.OLRcld[0],
              'run_date': rrtmg.creation_date,
              'S0': rrtmg.S0,
              'Ts': rrtmg.Ts[0],
              'albedo': rrtmg.aldif,
              'coszen': np.degrees(np.arccos(rrtmg.coszen))
              }
    text1 = f"$S_0=${params['S0']} W/m2, $OLR=${params['OLR']:0.2f} W/m2, $ASR=${params['ASR']:0.2f} W/m2"
    text2 = f"$T_s=${params['Ts']:0.2f} K, $Sfc. Albedo=${params['albedo']:0.2f}, Sun Angle={params['coszen']:0.2f} deg"
    text3 = f"RRTMG Run Time: {params['run_date']}"

    return params, text1, text2, text3
    
def prep_RRTMG_absorbers(atmosphere):
    # Get trace gases
    h2ovmr = abs_species(atmosphere, 'H2O')
    o3vmr = abs_species(atmosphere, 'O3')
    co2vmr = abs_species(atmosphere, 'CO2')
    ch4vmr = abs_species(atmosphere, 'CH4')
    n2ovmr = abs_species(atmosphere, 'N2O')
    o2vmr = abs_species(atmosphere, 'O2')
    
    # Heavy molecules (chloroforocarbons)
    cfc11vmr = abs_species(atmosphere, "CFC11")
    cfc12vmr = abs_species(atmosphere, "CFC12")
    cfc22vmr = abs_species(atmosphere, "CFC22")
    ccl4vmr = abs_species(atmosphere, "CCL4")

    # Set up the dictionary containing the gas absorber species concentrations
    absorber_vmr = {'CO2':co2vmr,
                    'CH4':ch4vmr,
                    'N2O':n2ovmr,
                    'O2':o2vmr,
                    'CFC11':cfc11vmr,
                    'CFC12':cfc12vmr,
                    'CFC22':cfc22vmr,
                    'CCL4':ccl4vmr,
                    'O3':o3vmr,}

    return absorber_vmr, h2ovmr

def prep_csv4rrtmg(atmosphere):
    
    # Pull out the trace gas concentrations, pressure, and temperature
    atmosphere = atmosphere.loc[:,['z', 't', 'p', 'H2O', 'O3', 'N2O', 'O2', 'CO2', 'CH4']]
    atmosphere.loc[:,['H2O', 'O3', 'N2O', 'O2', 'CO2', 'CH4']] = atmosphere.loc[:,['H2O', 'O3', 'N2O', 'O2', 'CO2', 'CH4']]/100.
    
    # heavy molecules
    atmosphere['CFC11'] = 0
    atmosphere['CFC12'] = 0
    atmosphere['CFC22'] = 0
    atmosphere['CCL4'] = 0
    
    atmosphere['Cloud_IWP'] = 0 # Ice Cloud Water Path
    atmosphere['Cloud_LWP'] = 0 # Liquid Cloud Water Path
    atmosphere['R_eff_ice'] = 0 # Ice effective radius
    atmosphere['R_eff_liq'] = 0 # Liquid effective radius
    
    # 
    atmosphere['SW_Cloud_Tau'] = 0 # In-cloud SW cloud optical depth (tauc_sw)
    atmosphere['SW_Cloud_SSA'] = 0 # In-cloud SW single scatter albedo (ssac_sw)
    atmosphere['SW_Cloud_Asym'] = 0 # In-cloud SW asymmetry parameter (asmc_sw)
    atmosphere['SW_Cloud_FSF'] = 0 # In-cloud SW forward scattering fraction (fsfc_sw)
    atmosphere['SW_Aerosol_Tau'] = 0 # Aerosol SW optical depth (tauaer_sw)
    atmosphere['SW_Aerosol_SSA'] = 0 # Aerosol SW optical depth (tauaer_sw)
    atmosphere['SW_Aerosol_Asym'] = 0 # Aerosol SW optical depth (tauaer_sw)
    atmosphere['LW_Cloud_Tau'] = 0 # In-cloud LW cloud optical depth (tauc_lw)
    atmosphere['LW_Aerosol_Tau'] = 0 # Aerosol LW optical depth (tauaer_lw)
    atmosphere['Cloud_Fraction'] = 0 # Cloud fraction

    return atmosphere

def prep_RRTMG_state(atmosphere):
    plev = atmosphere['p'].to_numpy()  # pressure bounds
    temperature = atmosphere['t'].to_numpy() # Temperature bounds
    tsfc = temperature[0]
    tlev = temperature
    
    # Set up the state variable with our temperatures and pressures.
    state = climlab.column_state(lev=plev.squeeze()/100., num_lat=1) # put the pressure array into the state
    state['Tatm'] = field.Field(tlev.squeeze(), domain=state['Tatm'].domain) # put temperature array into the state
    state['Ts'] = field.Field(tlev.squeeze()[0], domain=state['Ts'].domain) # put temperature array into the state

    return state

def prep_RRTMG_cloud(atmosphere, param=False):
    # Cloud Properties Arrays (Water path and effective radius)
    clwp = abs_species(atmosphere, "Cloud_LWP")  # in-cloud liquid water path (g/m2)
    ciwp = abs_species(atmosphere, "Cloud_IWP") # in-cloud ice water path (g/m2)
    r_liq = abs_species(atmosphere, "R_eff_liq") # Cloud water drop effective radius (microns) (2.5-60 micrometer limits)
    r_ice = abs_species(atmosphere, "R_eff_ice")   # Cloud ice particle effective size (microns)
    cldfrac = abs_species(atmosphere, "Cloud_Fraction")  # Cloud fraction

    tauc_lw = abs_species(atmosphere, "LW_Cloud_Tau") # Longwave liquid cloud optical depth 
    tauc_sw = abs_species(atmosphere, "SW_Cloud_Tau") # Shortwave liquid cloud optical depth
    ssac_sw = abs_species(atmosphere, "SW_Cloud_SSA") # In-cloud SW single scatter albedo (ssac_sw)
    asmc_sw = abs_species(atmosphere, "SW_Cloud_Asym") # In-cloud SW asymmetry parameter (asmc_sw)
    fsfc_sw = abs_species(atmosphere, "SW_Cloud_FSF") # In-cloud SW forward scattering fraction (fsfc_sw)
    
    if param is True:

        # Flag for cloud optical properties (inflglw, inflgsw)
            # INFLAG = 0 direct specification of optical depths of clouds;
            #            cloud fraction and cloud optical depth (gray) are
            #            input for each cloudy layer
            #        = 1 calculation of combined ice and liquid cloud optical depths (gray)
            #            as in CCM2; cloud fraction and cloud water path are input for
            #            each cloudy layer.
            #        = 2 calculation of separate ice and liquid cloud optical depths, with
            #            parameterizations determined by values of ICEFLAG and LIQFLAG.
            #            Cloud fraction, cloud water path, cloud ice fraction, and
            #            effective ice radius are input for each cloudy layer for all
            #            parameterizations.  If LIQFLAG = 1, effective liquid droplet radius
            #            is also needed.
        inflglw=2
        inflgsw=2
    
        # Flag for ice particle specification: (iceflglw, iceflgsw)
            #             ICEFLAG = 0 the optical depths (gray) due to ice clouds are computed as in CCM3.
            #                     = 1 the optical depths (non-gray) due to ice clouds are computed as closely as
            #                         possible to the method in E.E. Ebert and J.A. Curry, JGR, 97, 3831-3836 (1992).
            #                     = 2 the optical depths (non-gray) due to ice clouds are computed by a method
            #                         based on the parameterization used in the radiative transfer model Streamer
            #                         (reference,  J. Key, Streamer User's Guide, Technical Report 96-01] Boston
            #                         University, 85 pp. (1996)), which is closely related to the parameterization
            #                         of water clouds due to Hu and Stamnes (see below).
            #             = 3 the optical depths (non-gray) due to ice clouds are computed by a method
            # based on the parameterization given in Fu et al., J. Clim.,11,2223-2237 (1998).
            # specific definition of reic depends on setting of iceflglw:
            # iceflglw = 0,  ice effective radius, r_ec, (Ebert and Curry, 1992)]
            #               r_ec must be >= 10.0 microns
            # iceflglw = 1,  ice effective radius, r_ec, (Ebert and Curry, 1992)]
            #               r_ec range is limited to 13.0 to 130.0 microns
            # iceflglw = 2,  ice effective radius, r_k, (Key, Streamer Ref. Manual] 1996)
            #               r_k range is limited to 5.0 to 131.0 microns
            # iceflglw = 3,  generalized effective size, dge, (Fu, 1996)]
            #               dge range is limited to 5.0 to 140.0 microns
            #               [dge = 1.0315 * r_ec]
        iceflgsw=2
        iceflglw=1
    
        # Flag for liquid droplet specification: (liqflglw, liqflgsw)
            # LIQFLAG = 0 the optical depths (gray) due to water clouds are computed as in CCM3.
            #         = 1 the optical depths (non-gray) due to water clouds are computed by a method
            #             based on the parameterization of water clouds due to Y.X. Hu and K. Stamnes,
            #             J. Clim., 6, 728-742 (1993).
        liqflgsw=2
        liqflglw=1

        mycloud = {'clwp': clwp, 'ciwp': ciwp, 'r_liq': r_liq, 'r_ice': r_ice, 'cldfrac': cldfrac}
    else:
        inflglw=0
        inflgsw=0
        iceflgsw=0
        iceflglw=0
        liqflgsw=0
        liqflglw=0
        mycloud = {'tauc_lw': tauc_lw, 'tauc_sw': tauc_sw, 'ssac_sw': ssac_sw, \
                   'asmc_sw': asmc_sw, 'cldfrac': cldfrac, 'fsfc_sw': fsfc_sw}
        
    mycloud.update({'inflglw': inflglw, 'inflgsw': inflgsw, 'iceflgsw': iceflgsw, 'liqflgsw': liqflgsw, 'iceflglw': iceflglw,
                    'liqflglw': liqflglw})
        
    return mycloud
    
def prep_RRTMG_aerosol(atmosphere):
    # Initialize Cloud Properties Arrays
    tauaer_sw = abs_species(atmosphere, "SW_Aerosol_Tau")  # SW Aerosol optical depth (iaer=10 only)
    ssaaer_sw = abs_species(atmosphere, "SW_Aerosol_SSA") # SW Aerosol single scattering albedo
    asmaer_sw = abs_species(atmosphere, "SW_Aerosol_Asym") # SW Aerosol Asymmetry Parameter
    tauaer_lw = abs_species(atmosphere, "LW_Aerosol_Tau") # LW Aerosol optical depth
    iaer = 10 # Don't parameterize the radiative properties of the aerosol, use direct RT-relevant variables
    
    return {'tauaer_sw': tauaer_sw, 'ssaaer_sw': ssaaer_sw, 'asmaer_sw': asmaer_sw, 'tauaer_lw': tauaer_lw, 'iaer': iaer}

def plot_RRTMG(rrtmg, atmosphere, pressure_limits, hr_limits, flux_limits, net_flux_limits):
    plev = atmosphere['p'].to_numpy()  # pressure bounds

    dTdt_LW = rrtmg.TdotLW.to_xarray()
    LW_flux_up = rrtmg.LW_flux_up.to_xarray()
    LW_flux_down = rrtmg.LW_flux_down.to_xarray()
    
    dTdt_SW = rrtmg.TdotSW.to_xarray()
    SW_flux_up = rrtmg.SW_flux_up.to_xarray()
    SW_flux_down = rrtmg.SW_flux_down.to_xarray()
        
    params, text1, text2, text3 = get_rrtmg_params(rrtmg)

    fig, axs = pplt.subplots(ncols=3, nrows=1, figsize=(12,5), sharex=False, sharey=False)
    fig.format(ylim=pressure_limits, ylabel="Pressure [mb]")
    
    axs[0].plot(dTdt_LW, dTdt_LW.lev, label="LW Heating", color='k')
    axs[0].plot(dTdt_SW, dTdt_SW.lev, label="SW Heating", color='b')
    axs[0].plot(dTdt_SW + dTdt_LW, dTdt_SW.lev, label="Net Heating", color='r')
    
    axs[0].format(yscale='log', yreverse=True, xlabel="Heating Rate [K/day]", 
                  title="Heating/Cooling Rates", xlim=hr_limits)
    axs[0].axvline(x=0, color='k', alpha=0.5)
    axs[0].legend(ncols=1, loc='ur')
    axs[0].text(0, -0.2, text1, transform='axes')
    
    #axs[1].plot(rrtmg.LW_flux_up, LW_flux_up.lev_bounds, label="LW Flux Up", linestyle='--', color='k')
    #axs[1].plot(-rrtmg.LW_flux_down, LW_flux_down.lev_bounds, label="LW Flux Down", color='k')
    #axs[1].plot(rrtmg.SW_flux_up, SW_flux_up.lev_bounds, label="SW Flux Up", linestyle='--', color='b')
    #axs[1].plot(-rrtmg.SW_flux_down, SW_flux_down.lev_bounds, label="SW Flux Down", color='b')
    #axs[1].format(yscale='log', yreverse=True, xlabel="Flux [$W/m^2$]", title="Shortwave/Longwave Fluxes",
     #            xlim=flux_limits)
    #axs[1].legend(ncols=1, loc='ur')
    #axs[1].text(0, -0.2, text2, transform='axes')
    axs[1].plot(rrtmg.LW_flux_up, LW_flux_up.lev_bounds, label="LW Flux Up", linestyle='--', color='k')
    axs[1].plot(-rrtmg.LW_flux_down, LW_flux_down.lev_bounds, label="LW Flux Down", color='k')
    axs[1].plot(-rrtmg.LW_flux_net, LW_flux_up.lev_bounds, label="LW Net Flux", color='k', linestyle=':')
    axs[1].format(yscale='log', yreverse=True, xlabel="Flux [$W/m^2$]", title="Longwave Fluxes",
                 xlim=flux_limits)
    axs[1].legend(ncols=1, loc='ur')
    axs[1].text(0, -0.2, text2, transform='axes')
    axs[1].axvline(x=0, color='k', alpha=0.5)

    
    #axs[2].plot(rrtmg.LW_flux_net, LW_flux_up.lev_bounds, label="LW $F_{net}$", color='k')
    #axs[2].plot(rrtmg.SW_flux_net, SW_flux_up.lev_bounds, label="SW $F_{net}$", color='b')
    #axs[2].plot(rrtmg.LW_flux_net_clr, LW_flux_up.lev_bounds, label="LW $F_{net}$ Clear", color='k', linestyle=':')
    #axs[2].plot(rrtmg.SW_flux_net_clr, SW_flux_up.lev_bounds, label="SW $F_{net}$ Clear", color='b', linestyle=':')
    #axs[2].format(yscale='log', yreverse=True, xlabel="Flux [$W/m^2$]", title="Shortwave/Longwave Net Fluxes",
    #             xlim=net_flux_limits)
    #axs[2].legend(ncols=1, loc='ur')
    axs[2].plot(rrtmg.SW_flux_up, SW_flux_up.lev_bounds, label="SW Flux Up", linestyle='--', color='b')
    axs[2].plot(-rrtmg.SW_flux_down, SW_flux_down.lev_bounds, label="SW Flux Down", color='b')
    axs[2].plot(-rrtmg.SW_flux_net, SW_flux_up.lev_bounds, label="SW Net Flux", color='b', linestyle=':')
    axs[2].format(yscale='log', yreverse=True, xlabel="Flux [$W/m^2$]", title="Shortwave Fluxes",
                 xlim=net_flux_limits)
    axs[2].axvline(x=0, color='k', alpha=0.5)

    axs[2].legend(ncols=1, loc='ur')
    axs[2].text(0, -0.2, text3, transform='axes')
    
    pplt.show()

def plot_OLR_Spectra(rrtmg):
    olr_spectral = rrtmg.OLR_spectral.to_xarray()
    wavenumbers = np.linspace(0.1, 3000) # don't start from zero to avoid divide by zero warnings
    
    # Centers and Widths of the spectral bands, cm-1
    spectral_centers = rrtmg.OLR_spectral.domain.axes['wavenumber'].points
    spectral_widths = rrtmg.OLR_spectral.domain.axes['wavenumber'].delta
    
    def planck_curve(wavenumber, T):
        '''Return the Planck curve in units of W/m2/cm-1
        Inputs: wavenumber in cm-1
                temperature T in units of K'''
    
        # 100pi factor converts from steradians/m to 1/cm
        return (climlab.utils.thermo.Planck_wavenumber(wavenumber, T)*100*np.pi)
    
    def make_planck_curve(ax, T, color='orange'):
        '''Plot the Planck curve (W/m2/cm-1) on the given ax object'''
        ax.plot(wavenumbers, planck_curve(wavenumbers, T),
                lw=2, color=color, label="Planck Curve, {}K".format(T))
    
    def make_rrtmg_spectrum(ax, OLR_spectral, color='blue', alpha=0.5, label='RRTMG - 300K'):
        # Need to normalize RRTMG spectral outputs by width of each wavenumber band
        ax.bar(spectral_centers, np.squeeze(OLR_spectral)/spectral_widths,
               width=spectral_widths, color=color, edgecolor='black', alpha=alpha, label=label)
        
    fig, axs = plt.subplots(figsize=(7,4))
    make_rrtmg_spectrum(axs, olr_spectral.to_numpy().squeeze(), label='RRTMG')
    make_planck_curve(axs, 268, color='orange')
    axs.legend(frameon=False)
    axs.set_xlabel("Wavenumber [cm$^{-1}$]")
    axs.set_ylabel("TOA Flux [W/m$^{2}$/cm$^{-1}$]")
    plt.show()