"""Calculate and plot absorption cross sections."""
import re

import numpy as np
import pyarts
import scipy as sp

def getLines(species="N2O",
             fmin=10e9,
             fmax=2000e9,
             fnum=10_000,
             cut_off=1e-2):
    
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
    
    line_location = []
    line_strength = []
    for abslines in ws.abs_lines_per_species.value[0]:
        for line in abslines.lines:
            line_location.append(line.F0)
            line_strength.append(line.I0)
    print(abslines.meta_data)
    line_location = np.asarray(line_location)
    line_strength = np.asarray(line_strength)
    
    return line_location, line_strength
