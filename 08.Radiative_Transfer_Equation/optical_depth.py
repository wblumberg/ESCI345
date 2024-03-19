import pyarts.workspace
import pandas as pd
import numpy as np
import pint
import os
import xarray as xr
import pint_xarray

ureg = pint.get_application_registry()

# Default frequency grid is for the microwave used in a lot of RT applications.
default_fmin = 10e9 * ureg.Unit("Hz")
default_fmax = 200e9 * ureg.Unit("Hz")

mw_species = {"O2" : 'O2-*, O2-SelfContStandardType', \
              "H2O": 'H2O-*, H2O-SelfContStandardType, H2O-ForeignContStandardType', \
              "N2": 'N2-*, N2-SelfContStandardType', \
              "O3": "O3-*"}

ir_species = {"H2O": 'H2O, H2O-SelfContCKDMT400, H2O-ForeignContCKDMT400', \
              "CO2": 'CO2-*', \
              "O3": 'O3-*', \
              "N2O": 'N2O-*', \
              "CO": 'CO-*', \
              "CH4": 'CH4-*', \
              "O2": 'O2-*'}

arts_xml_atmospheres = os.environ.get('ARTS_XML_ATMO') 

def getMicrowaveAbsorbers():
    return mw_species

def getInfraredAbsorbers():
    return ir_species

def atmo_optical_depth(fbounds=(default_fmin, default_fmax), fnum=1_000, abs_species=mw_species, atmosphere='tropical', linesoff=False):
    # Create a list of all the absorbers:
    gas_absorbers = []
    for key in abs_species.keys():
        gas_absorbers.append(abs_species[key])
    
    # Save the units we used 
    original_units = fbounds[0].units

    # Convert our spectral grid to the frequency unit (since that's what PyARTS uses)
    fmin = fbounds[0].to("Hz", "spectroscopy").magnitude
    fmax = fbounds[1].to("Hz", "spectroscopy").magnitude

    # Set up the PyARTS Workspace
    ws = pyarts.workspace.Workspace(verbosity=0)
    ws.water_p_eq_agendaSet()
    ws.PlanetSet(option="Earth") # Set this to be on Earth
    ws.ppath_agendaSet(option="PlaneParallel") # Set this to be a plane-parallel atmosphere
    ws.gas_scatteringOff() # No scattering by the atmospheric gases (pure absorption)

    # Number of Stokes components to be computed
    ws.IndexSet(ws.stokes_dim, 1)

    # No jacobian calculation
    ws.jacobianOff()

    # Clearsky = No scattering
    ws.cloudboxOff()

    # Set the gas absorbers that we've specified
    ws.abs_speciesSet(species=gas_absorbers)

    # Read in the HITRAN XML Line files
    ws.abs_lines_per_speciesReadSpeciesSplitCatalog(basename='lines/')
    ws.abs_lines_per_speciesCutoff(option="ByLine", value=750e9)
    
    # Load CKDMT400 model data and continuum absorption data and HITRAN XSEC
    ws.ReadXML(ws.predefined_model_data, "model/mt_ckd_4.0/H2O.xml")

    # Read in the CIA Data
    ws.abs_cia_dataReadSpeciesSplitCatalog(basename="cia/")

    # Read in the HITRAN Experimentally Determine Cross-Section Data (not doing anything with this currently)
    ws.ReadXsecData(basename="xsec/")
    ws.abs_lines_per_speciesTurnOffLineMixing()

    # Create our frequency grid
    ws.VectorNLinSpace(ws.f_grid, int(fnum), float(fmin), float(fmax))

    # Create our temporary pressure grid (50 level atmosphere)
    ws.VectorNLogSpace(ws.p_grid, 50, 1013e2, 10000.0)
    
    if linesoff:
        ws.abs_lines_per_speciesSetEmpty()

    # Throw away lines outside f_grid
    ws.abs_lines_per_speciesCompact()

    # Let's specify an FASCODE atmosphere (e.g., tropical, midlatitude-summer, etc.) to set 
    # the atmospheric properties (temperature, pressure, height, vmr for absorbing gases).
    if type(atmosphere) == pd.DataFrame:
        ws.AtmRawRead(basename=f"{arts_xml_atmospheres}/midlatitude-summer/midlatitude-summer")
        atmosphere_type = "CUSTOM"
        ws.VectorNLogSpace(ws.p_grid, len(atmosphere['p']), 1013e2, 10000.0)
    elif type(atmosphere) == str:
        # even though we have a custom atmosphere, we need this line to initialize the variables t_field_raw, z_field_raw, and vmr_field_raw.  I'm not sure how to do this otherwise.
        ws.AtmRawRead(basename=f"{arts_xml_atmospheres}/{atmosphere}/{atmosphere}")
        atmosphere_type = "FASCODE"
    else:
        print("Invalid atmosphere specified:", type(atmosphere))
        return
        
    ws.sensorOff() # No sensor properties
    ws.propmat_clearsky_agendaAuto() # Set this to be a clear-sky scenario
    ws.AtmosphereSet1D() # Set it to be a 1D Column
    ws.AtmFieldsCalc() # Interpolate the atmosphere using the pressure grid specified.  This will create t_field, z_field, and vmr_field

    # Now, if we're using a custom atmosphere profile, let's overwrite the profiles in the PyARTS variables.
    if atmosphere_type == "CUSTOM":
        ws.p_grid = atmosphere['p'].to_numpy()
        ws.t_field = atmosphere['t'].to_numpy().reshape((len(atmosphere['p'].to_numpy()),1,1))
        ws.z_field = atmosphere['z'].to_numpy().reshape((len(atmosphere['p'].to_numpy()),1,1))
        for i, gas_s in enumerate(abs_species.keys()):
            ws.vmr_field.value[i] = atmosphere[gas_s].to_numpy().reshape((len(atmosphere['p'].to_numpy()),1,1))

    ws.atmfields_checkedCalc() # Check the fields for this atmosphere
    ws.lbl_checkedCalc() # Check to see if we can do a line-by-line calculation
    
    # https://atmtools.github.io/arts-docs-master/docserver/variables/propmat_clearsky_field.html
    ws.propmat_clearsky_fieldCalc() # Calculate the volumetric absorption coefficients (m-1).

    # Get the absorption coefficient matrix so we can calculate Optical Depths
    abs_coeff_matrix = ws.propmat_clearsky_field.value.value.squeeze() # squeeze will reduce this to a [species, f_grid, p_grid] array

    # Let's get the atmospheric profile variables put together
    atmo_profile = {"Temperature": ws.t_field.value.value.squeeze(),
                    "Height": ws.z_field.value.value.squeeze(),
                    "Pressure": ws.p_grid.value.value.squeeze()}

    # Iterate over every absorber specified and save the profile we used.
    for i, gas in enumerate(abs_species.keys()):
        atmo_profile[gas] = ws.vmr_field.value[i][:,0,0]

    height = atmo_profile["Height"]
    dz = np.diff(height) # Calculate the dz so we can compute optical depths.

    # Save the atmospheric profiles as a DataFrame so we can use this again.
    atmo_profile = pd.DataFrame(atmo_profile)
    atmo_profile.name ="Atmospheric Profiles"

    # Let's compute the optical depths:
    abs_coeff_shape = abs_coeff_matrix.shape
    optical_depths = np.zeros((abs_coeff_matrix.shape[0], abs_coeff_matrix.shape[1], abs_coeff_matrix.shape[2] - 1))
    for s in range(len(abs_coeff_matrix)): # iterate over each species
        for i in range(1,len(dz)): # iterate over every layer
            optical_depths[s,:,i-1] = np.trapz(y=abs_coeff_matrix[s,:,i-1:i+1], x=height[i-1:i+1], axis=1)

    # Compute some the individual absorber transmission spectra and the total optical depth of each absorber.
    total_transmission = pd.DataFrame(np.exp(-np.sum(optical_depths, axis=2)), index=abs_species.keys())
    total_transmission.name = "Gas Transmission"
    total_optical_depth = pd.DataFrame(np.sum(optical_depths, axis=2), index=abs_species.keys())
    total_optical_depth.name = "Gas Optical Depths"
    all_optical_depth = pd.Series(np.sum(total_optical_depth, axis=0), name="Total Optical Depth")

    # Copy results from ARTS
    frequency_grid = ws.f_grid.value.value.copy()
    
    # Add units from pint to result
    frequency_grid = np.asarray(frequency_grid) * ureg.Unit("Hz")

    # Convert to the original "frequency" units.
    frequency_grid = frequency_grid.to(original_units, "spectroscopy")

    avg_h = (height[1:] + height[:-1])/2.
    # Use the power of xarray, young one.
    optical_depths = xr.DataArray(optical_depths, dims=['gas_species','spectral_unit','layer_center'],
                                  coords={'gas_species': np.asarray(list(abs_species.keys())),
                                          'spectral_unit': frequency_grid,
                                          'layer_center': avg_h})
    optical_depths = optical_depths.pint.quantify({'spectral_unit': frequency_grid.units, 'layer_center': pint.Unit('m')})
    
    return atmo_profile, optical_depths, frequency_grid

def getTotalGasTransmission(optical_depths):
    total_transmission = np.exp(-np.sum(optical_depths, axis=2))
    total_transmission.name = "Gas Transmission"

    return total_transmission

def getTotalGasOpticalDepth(optical_depths):
    total_optical_depth = np.sum(optical_depths, axis=2)
    total_optical_depth.name = "Gas Optical Depths"
    return total_optical_depth

def mw_optical_depth(fmin=10e9,fmax=250e9,fnum=1_000,atmosphere="midlatitude-summer", atmosphere_path = '/home/jovyan/ESCI345/', linesoff=False):
    ws = pyarts.workspace.Workspace(verbosity=0)
    ws.water_p_eq_agendaSet()
    ws.PlanetSet(option="Earth")
    ws.ppath_agendaSet(option="PlaneParallel")
    ws.gas_scatteringOff()

    # Number of Stokes components to be computed
    ws.IndexSet(ws.stokes_dim, 1)

    # No jacobian calculation
    ws.jacobianOff()

    # Clearsky = No scattering
    ws.cloudboxOff()

    # A pressure grid rougly matching 0 to 80 km, in steps of 2 km.
    ws.VectorNLogSpace(ws.p_grid, 50, 1013e2, 10000.0)

    ws.abs_speciesSet(species=['O2, O2-SelfContStandardType', \
                               'H2O, H2O-SelfContStandardType, H2O-ForeignContStandardType', \
                               'N2, N2-SelfContStandardType'])

    # Read a line file and a matching small frequency grid
    ws.abs_lines_per_speciesReadSpeciesSplitCatalog(basename="lines/")
    ws.abs_cia_dataReadSpeciesSplitCatalog(basename="cia/")
    ws.abs_lines_per_speciesCutoff(option="ByLine", value=750e9)
    ws.abs_lines_per_speciesTurnOffLineMixing()

    # Create a frequency grid
    ws.VectorNLinSpace(ws.f_grid, int(fnum), float(fmin), float(fmax))

    if linesoff:
        ws.abs_lines_per_speciesSetEmpty()

    # Throw away lines outside f_grid
    ws.abs_lines_per_speciesCompact()

    # Atmospheric scenario
    ws.AtmRawRead(basename=f"{atmosphere_path}/atmospheres/{atmosphere}/{atmosphere}")

    # No sensor properties
    ws.sensorOff()
    ws.propmat_clearsky_agendaAuto()
    ws.AtmosphereSet1D() # Set it to be a 1D Column
    ws.AtmFieldsCalc() # Interpolate the atmosphere using the pressure grid specified.

    ws.atmfields_checkedCalc()
    ws.lbl_checkedCalc()
    
    # https://atmtools.github.io/arts-docs-master/docserver/variables/propmat_clearsky_field.html
    ws.propmat_clearsky_fieldCalc()
    abs_coeff_matrix = ws.propmat_clearsky_field.value.value.squeeze() # squeeze will reduce this to a [species, f_grid, p_grid] array
    
    temperature = ws.t_field.value.value.squeeze()
    height = ws.z_field.value.value.squeeze()
    vmr_o2 = ws.vmr_field.value[0][:,0,0] # p_grid, lat_grid, lon_grid
    vmr_h2o = ws.vmr_field.value[1][:,0,0]
    vmr_n2 = ws.vmr_field.value[2][:,0,0]
    dz = np.diff(height)

    profiles = pd.DataFrame({'temperature': temperature,
                             'height': height,
                             'vmr_o2': vmr_o2,
                             'vmr_h2o': vmr_h2o,
                             'vmr_n2': vmr_n2})
    profiles.name ="Atmospheric Profiles"
    abs_coeff_shape = abs_coeff_matrix.shape

    optical_depths = np.zeros((abs_coeff_matrix.shape[0], abs_coeff_matrix.shape[1], abs_coeff_matrix.shape[2] - 1))

    for s in range(len(abs_coeff_matrix)):
        for i in range(1,len(dz)):
            optical_depths[s,:,i-1] = np.trapz(y=abs_coeff_matrix[s,:,i-1:i+1], x=height[i-1:i+1], axis=1)

    freq = ws.f_grid.value.value / 1e9 # Convert to GHz
                            
    total_transmission = pd.DataFrame(np.exp(-np.sum(optical_depths, axis=2)), index=['O2',"H2O","N2"])
    total_transmission.name = "Gas Transmission"
    total_optical_depth = pd.DataFrame(np.sum(optical_depths, axis=2), index=['O2',"H2O","N2"])
    total_optical_depth.name = "Gas Optical Depths"
    all_optical_depth = pd.Series(np.sum(total_optical_depth, axis=0), name="Total Optical Depth")
    
    return profiles, optical_depths, total_transmission, total_optical_depth, all_optical_depth, freq

def ir_optical_depth(wlen_min=3, wlen_max=20, atmosphere="midlatitude-summer", atmosphere_path = '/home/jovyan/ESCI345/', linesoff=False):
    wlen_min = ureg.Quantity(wlen_min, units='micrometers')
    wlen_max = ureg.Quantity(wlen_max, units='micrometers')
    fmax = wlen_min.to('Hz', 'spectroscopy')
    fmin = wlen_max.to('Hz', 'spectroscopy')

    fnum=10_000
    fmin=fmin.magnitude
    fmax=fmax.magnitude
    atmosphere_path="./"

    ws = pyarts.workspace.Workspace(verbosity=0)
    ws.water_p_eq_agendaSet()
    ws.PlanetSet(option="Earth")
    #ws.iy_main_agendaSet(option="EmissionPlaneParallel")
    ws.ppath_agendaSet(option="PlaneParallel")
    ws.gas_scatteringOff()

    # Number of Stokes components to be computed
    ws.IndexSet(ws.stokes_dim, 1)

    # No jacobian calculation
    ws.jacobianOff()

    # Clearsky = No scattering
    ws.cloudboxOff()

    # A pressure grid rougly matching 0 to 80 km, in steps of 2 km.
    ws.VectorNLogSpace(ws.p_grid, 40, 1013e2, 10000)

    ws.abs_speciesSet(species=['H2O', \
                               'CO2', \
                               'CH4', \
                               'O3', \
                               'N2O'])

    # Read a line file and a matching small frequency grid
    ws.abs_lines_per_speciesReadSpeciesSplitCatalog(basename="lines/")
    ws.abs_lines_per_speciesCutoff(option="ByLine", value=750e9)
    ws.abs_lines_per_speciesTurnOffLineMixing()

    # Create a frequency grid
    ws.VectorNLinSpace(ws.f_grid, int(fnum), float(fmin), float(fmax))

    if linesoff:
        ws.abs_lines_per_speciesSetEmpty()

    # Throw away lines outside f_grid
    ws.abs_lines_per_speciesCompact()

    # Atmospheric scenario
    ws.AtmRawRead(basename=f"{atmosphere_path}/atmospheres/{atmosphere}/{atmosphere}")

    # Non reflecting surface
    #ws.VectorSetConstant(ws.surface_scalar_reflectivity, 1, 0.1)
    #ws.surface_rtprop_agendaSet(
    #    option="Specular_NoPol_ReflFix_SurfTFromt_surface")

    # No sensor properties
    ws.sensorOff()
    ws.propmat_clearsky_agendaAuto()
    ws.AtmosphereSet1D() # Set it to be a 1D Column
    ws.AtmFieldsCalc() # Interpolate the atmosphere using the pressure grid specified.

    ws.atmfields_checkedCalc()
    ws.lbl_checkedCalc()
    
    temperature = ws.t_field.value.value.squeeze()
    height = ws.z_field.value.value.squeeze()
    vmr_h2o = ws.vmr_field.value[0][:,0,0]
    vmr_co2 = ws.vmr_field.value[1][:,0,0]
    vmr_ch4 = ws.vmr_field.value[2][:,0,0]
    vmr_o3 = ws.vmr_field.value[3][:,0,0]
    vmr_no2 = ws.vmr_field.value[4][:,0,0]
    
    profiles = pd.DataFrame({'temperature': temperature,
                             'height': height,
                             'vmr_h2o': vmr_h2o,
                             'vmr_co2': vmr_co2,
                             'vmr_ch4': vmr_ch4,
                             'vmr_o3': vmr_o3,
                             'vmr_n2o': vmr_no2})
    # https://atmtools.github.io/arts-docs-master/docserver/variables/propmat_clearsky_field.html
    ws.propmat_clearsky_fieldCalc()
    abs_coeff_matrix = ws.propmat_clearsky_field.value.value.squeeze() # squeeze will reduce this to a [species, f_grid, p_grid] array
    
    dz = np.diff(height)

    abs_coeff_shape = abs_coeff_matrix.shape

    optical_depths = np.zeros((abs_coeff_matrix.shape[0], abs_coeff_matrix.shape[1], abs_coeff_matrix.shape[2] - 1))

    for s in range(len(abs_coeff_matrix)):
        for i in range(1,len(dz)):
            optical_depths[s,:,i-1] = np.trapz(y=abs_coeff_matrix[s,:,i-1:i+1], x=height[i-1:i+1], axis=1)

    total_transmission = np.exp(-np.sum(optical_depths, axis=2))
    total_optical_depth = np.sum(optical_depths, axis=2)
    all_optical_depth = np.sum(total_optical_depth, axis=0)

    total_transmission = pd.DataFrame(np.exp(-np.sum(optical_depths, axis=2)), index=['H2O',"CO2","CH4","O3","N2O"])
    total_transmission.name = "Gas Transmission"
    total_optical_depth = pd.DataFrame(np.sum(optical_depths, axis=2), index=['H2O',"CO2","CH4","O3","N2O"])
    total_optical_depth.name = "Gas Optical Depths"
    all_optical_depth = pd.Series(np.sum(total_optical_depth, axis=0), name="Total Optical Depth")
    
    wlen_grid = (ws.f_grid.value.value * ureg.Hz).to('micrometer', 'spectroscopy')
    wlen_grid = wlen_grid.magnitude
    
    return profiles, optical_depths, total_transmission, total_optical_depth, all_optical_depth, wlen_grid
