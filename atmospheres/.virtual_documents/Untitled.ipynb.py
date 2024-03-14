import pandas as pd
import glob
import numpy as np
import proplot as pplt
pplt.use_style('seaborn-poster')
pplt.rc["legend.facecolor"] = "white"


files = glob.glob("subarctic-winter//*.*.xml")


data_dict = {}
for f in files:
    data = pd.read_xml(f)
    data = np.asarray(data["Tensor3"][0].split('\n'), dtype=float)
    if len(data) > 47:
        continue
    
    var = f.split('/')[1].split('.')[1]
    if var get_ipython().getoutput("= "z":")
        data = data * 100.
    print(data.shape, var)
    data_dict[var] = data[1:]
data = pd.DataFrame(data_dict)
data['z'] = data['z']/1000.
data = data.set_index('z')
data


fig, axs = pplt.subplots(ncols=1, nrows=1, figsize=(12,9))
axs[0].plot(data["N2"], data.index, label=r"$N_2$")
axs[0].plot(data["O2"], data.index, label=r"$O_2$")
axs[0].plot(data["H2O"], data.index, label=r"$H_2 O$")
axs[0].plot(data["CH4"], data.index, label=r"$CH_4$")
axs[0].plot(data["CO2"], data.index, label=r"$CO_2$")
axs[0].plot(data["NO2"], data.index, label=r"$NO_2$")
axs[0].plot(data["O3"], data.index, label=r"$O_3$")

axs.format(xlim=(1e-6,100), xscale='log', ylim=(0,100), ylabel="Height [km AGL]", xlabel="Concentration by Volume [get_ipython().run_line_magic("]",", " grid=True)")
fig.legend(loc='bottom', ncols=9)
fig.format(title="Subarctic Winter Atmosphere")


data["N2"].to_numpy(), data.index



