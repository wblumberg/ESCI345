"""Calculate and plot absorption related variables."""

import numpy as np
import pyarts
import scipy as sp
import pint
import re
import os

molecular_weights = {"NH3": 17.03,
                    "Ar": 39.95,
                    "He": 4.00,
                    "H2": 2.02,
                    "H2S": 34.08,
                    "Kr": 83.80,
                    "CH4": 16.04,
                    "Ne": 20.18,
                    "NO": 30.01,
                    "N2": 28.01,
                    "NO2": 46.01,
                    "N2O": 44.01,
                    "O2": 32.00,
                    "O3": 48.00,
                    "CO2": 44.01,
                    "CO": 28.01,
                    "SO2": 64.06,
                    "Xe": 131.29,
                    "H2O": 18.02} # g/mol

def tag2tex(tag):
    """Replace all numbers in a species tag with LaTeX subscripts."""
    return re.sub("([a-zA-Z]+)([0-9]+)", r"\1$_{\2}$", tag)


ureg = pint.get_application_registry()

default_fmin = 10e9 * ureg.Unit("Hz")
default_fmax = 2000e9 * ureg.Unit("Hz")

####
#### Code to access spectral lines for specific isotopes in the HITRAN database
#### Returns the line position and line strength.
####
def getLines(species="N2O",
             fmin=default_fmin,
             fmax=default_fmax,
             fnum=10_000,
             cut_off=1e-2):
    """Pull from the HITRAN database ARTS and get spectral lines for a species.

    Parameters:
        species (str): Absorption species name.
        fmin (float): Minimum frequency (pint units).
        fmax (float): Maximum frequency (pint units).
        fnum (int): Number of frequency grid points.
        cut_off (float): Cut off frequency for lines.

    Returns:
        ndarray, ndarray: Line location [spectral units], Line strength [m2 Hz-1]
    """
    
    original_units = fmax.units
    
    fmin = fmin.to("Hz", "spectroscopy").magnitude
    fmax = fmax.to("Hz", "spectroscopy").magnitude
    
    ws = pyarts.workspace.Workspace(verbosity=0)
    ws.water_p_eq_agendaSet()
    ws.PlanetSet(option="Earth")

    ws.verbositySetScreen(ws.verbosity, 0)

    # We do not want to calculate the Jacobian Matrix
    ws.jacobianOff()

    # Define absorption species
    ws.abs_speciesSet(species=[species])

    ws.abs_lines_per_speciesReadSpeciesSplitCatalog(basename='lines/')
    ws.abs_lines_per_speciesCutoff(option="ByLine", value=cut_off)
    
    # Load CKDMT400 model data and continuum absorption data and HITRAN XSEC
    ws.ReadXML(ws.predefined_model_data, "model/mt_ckd_4.0/H2O.xml")
    ws.abs_cia_dataReadSpeciesSplitCatalog(basename="cia/")
    ws.ReadXsecData(basename="xsec/")
    
    # Create a frequency grid
    ws.VectorNLinSpace(ws.f_grid, fnum, fmin, fmax)

    # Throw away lines outside f_grid
    ws.abs_lines_per_speciesCompact()
   
    # Loop through the lines and get the individual line locations and their strength
    line_location = []
    line_strength = []
    for abslines in ws.abs_lines_per_species.value[0]:
        for line in abslines.lines:
            line_location.append(line.F0)
            line_strength.append(line.I0)
    
    print(abslines.meta_data)
    line_location = np.asarray(line_location) * ureg.Unit("Hz")
    line_strength = np.asarray(line_strength) * ureg.Unit("meter**2 Hz**-1")
    
    line_location = line_location.to(original_units, "spectroscopy")
    
    return line_location, line_strength


def calculate_absxsec(species="N2O",
                      pressure=800e2,
                      temperature=300.0,
                      fmin=default_fmin,
                      fmax=default_fmax,
                      fnum=10_000,
                      lineshape="LP",
                      normalization="RQ",
                      verbosity=0,
                      vmr=0.05,
                      lines_off=0):
    """Calculate absorption cross sections.

    Parameters:
        species (str): Absorption species name.
        pressure (float): Atmospheric pressure [Pa].
        temperature (float): Atmospheric temperature [K].
        fmin (float): Minimum frequency (pint units).
        fmax (float): Maximum frequency (pint units).
        fnum (int): Number of frequency grid points.
        lineshape (str): Line shape model.
                            Available options:
                            DP        -      Doppler profile,
                            LP        -      Lorentz profile,
                            VP        -      Voigt profile,
                            SDVP      -      Speed-dependent Voigt profile,
                            HTP       -      Hartman-Tran profile.
        normalization (str): Line shape normalization factor.
                            Available options:
                            VVH       -      Van Vleck and Huber,
                            VVW       -      Van Vleck and Weisskopf,
                            RQ        -      Rosenkranz quadratic,
                            None      -      No extra normalization.
        verbosity (int): Set ARTS verbosity (``0`` prevents all output).
        vmr (float): Volume mixing ratio. This is mainly important for the
                     water vapor continua.
        lines_off (int): Switch off lines, if no contnua is included in the species string,
                         absorption will be zero.

    Returns:
        ndarray, ndarray, ndarray: Frequency grid [Hz], Abs. coefficient [m-1], Abs. cross sections [m^2]
    """
    
    # Keep track of the original "frequency" units.
    original_units = fmax.units
    
    fmin = fmin.to("Hz", "spectroscopy").magnitude
    fmax = fmax.to("Hz", "spectroscopy").magnitude
    
    # Create ARTS workspace and load default settings
    ws = pyarts.workspace.Workspace(verbosity=0)
    ws.water_p_eq_agendaSet()
    ws.PlanetSet(option="Earth")

    ws.verbositySetScreen(ws.verbosity, verbosity)

    # We do not want to calculate the Jacobian Matrix
    ws.jacobianOff()

    # Define absorption species
    ws.abs_speciesSet(species=[species])
    ws.abs_lines_per_speciesReadSpeciesSplitCatalog(basename='lines/')
    ws.abs_lines_per_speciesLineShapeType(option=lineshape)
    #ws.abs_lines_per_speciesCutoff(option="ByLine", value=750e9)
    ws.abs_lines_per_speciesNormalization(option=normalization)
    
    # Load CKDMT400 model data and continuum absorption data and HITRAN XSEC
    ws.ReadXML(ws.predefined_model_data, os.environ["ARTS_DATA_PATH"]+"model/mt_ckd_4.0/H2O.xml")
    ws.abs_cia_dataReadSpeciesSplitCatalog(basename="cia/")
    ws.ReadXsecData(basename="xsec/")
   
    if lines_off:
        ws.abs_lines_per_speciesSetEmpty()

    # Create a frequency grid
    ws.VectorNLinSpace(ws.f_grid, fnum, fmin, fmax)

    # Throw away lines outside f_grid
    ws.abs_lines_per_speciesCompact()

    # Atmospheric settings
    ws.AtmosphereSet1D()
    ws.stokes_dim = 1

    # Setting the pressure, temperature and vmr of the gas
    ws.rtp_pressure = float(pressure)  # [Pa]
    ws.rtp_temperature = float(temperature)  # [K]
    ws.rtp_vmr = np.array([vmr])  # [VMR]
    ws.Touch(ws.rtp_nlte)

    # isotop information
    ws.isotopologue_ratiosInitFromBuiltin()

    # Calculate absorption cross sections
    ws.lbl_checkedCalc()
    ws.propmat_clearsky_agenda_checked = 1
    ws.propmat_clearskyInit()
    ws.propmat_clearskyAddLines()
    ws.propmat_clearskyAddPredefined()

    # Convert abs coeff to cross sections on return - need number density
    number_density = pressure * vmr / (pyarts.arts.constants.k * temperature)

    # Copy results from ARTS
    frequency_grid = ws.f_grid.value.value.copy()
    absorption_coefficient = ws.propmat_clearsky.value.data.value[0, 0, :, 0].copy() # m-1
    absorption_cross_section = ws.propmat_clearsky.value.data.value[0, 0, :, 0].copy() / number_density #m2
    
    # Add units from pint to result
    frequency_grid = np.asarray(frequency_grid) * ureg.Unit("Hz")
    absorption_coefficient = np.asarray(absorption_coefficient) * ureg.Unit("meter**-1")
    absorption_cross_section = np.asarray(absorption_cross_section) * ureg.Unit("meter**2")
    
    # Convert to the original "frequency" units.
    frequency_grid = frequency_grid.to(original_units, "spectroscopy")
    
    return (frequency_grid, absorption_coefficient, absorption_cross_section)




def linewidth(f, a):
    """Calculate the full-width at half maximum (FWHM) of an absorption line.

        Parameters:
            f (ndarray): Frequency grid.
            a (ndarray): Line properties
                (e.g. absorption coefficients or cross-sections).

        Returns:
            float: Linewidth.

        Examples:
            >>> f = np.linspace(0, np.pi, 100)
            >>> a = np.sin(f)**2
            >>> linewidth(f, a)
            1.571048056449009
    """

    idx = np.argmax(a)

    if idx < 3 or idx > len(a) - 3:
        raise RuntimeError('Maximum is located too near at the edge.\n' +
                           'Could not found any peak. \n' +
                           'Please adjust the frequency range.')

    s = sp.interpolate.UnivariateSpline(f, a - np.max(a) / 2, s=0)

    zeros = s.roots()
    sidx = np.argsort((zeros - f[idx])**2)

    if zeros.size == 2:

        logic = zeros[sidx] > f[idx]

        if np.sum(logic) == 1:

            fwhm = abs(np.diff(zeros[sidx])[0])

        else:

            print(
                'I only found one half maxima.\n' +
                'You should adjust the frequency range to have more reliable results.\n'
            )

            fwhm = abs(zeros[sidx[0]] - f[idx]) * 2

    elif zeros.size == 1:

        fwhm = abs(zeros[0] - f[idx]) * 2

        print(
            'I only found one half maxima.\n' +
            'You should adjust the frequency range to have more reliable results.\n'
        )

    elif zeros.size > 2:

        sidx = sidx[0:2]

        logic = zeros[sidx] > f[idx]

        print('It seems, that there are more than one peak' +
              ' within the frequency range.\n' +
              'I stick to the maximum peak.\n' +
              'But I would suggest to adjust the frequevncy range. \n')

        if np.sum(logic) == 1:

            fwhm = abs(np.diff(zeros[sidx])[0])

        else:

            print(
                'I only found one half maxima.\n' +
                'You should adjust the frequency range to have more reliable results.\n'
            )

            fwhm = abs(zeros[sidx[0]] - f[idx]) * 2

    elif zeros.size == 0:

        raise RuntimeError('Could not found any peak. :( \n' +
                           'Probably, frequency range is too small.\n')

    return fwhm


def get_spectral_unit(unit):
    if str(unit[0].dimensionality) == "[length]":
        xlabel = f"Wavelength [${unit[0].units:~P}$]"
    elif str(unit[0].dimensionality) == "1 / [time]":
        xlabel = f"Frequency [${unit[0].units:~P}$]"
    elif str(unit[0].dimensionality) == "1 / [length]":
        xlabel = f"Wavenumber [${unit[0].units:~P}$]"
    return xlabel
