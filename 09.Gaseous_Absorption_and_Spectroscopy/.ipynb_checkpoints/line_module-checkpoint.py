"""Calculate and plot absorption cross sections."""
import re

import numpy as np
import pyarts
import scipy as sp
import pint
import re

def tag2tex(tag):
    """Replace all numbers in a species tag with LaTeX subscripts."""
    return re.sub("([a-zA-Z]+)([0-9]+)", r"\1$_{\2}$", tag)


ureg = pint.get_application_registry()

default_fmin = 10e9 * ureg.Unit("Hz")
default_fmax = 2000e9 * ureg.Unit("Hz")

def getLines(species="N2O",
             fmin=default_fmin,
             fmax=default_fmax,
             fnum=10_000,
             cut_off=1e-2):
    
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
