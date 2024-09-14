import sys
sys.path.insert(0, './src')
from interface import ImDataParamsBMRR
import numpy as np

I = ImDataParamsBMRR("./data/sim/proc/20231205_165340_202_ImDataParamsBMRR_sim_geo.h5")
I.ImDataParams["voxelSize_mm"] = np.array([1.5, 1.5, 5.0])
I.ImDataParams["signal"] = I.ImDataParams["signal"][:, :, :, 0, :]
I.set_FatModel()
I.set_VARPROparams()
I.VARPROparams["range_fm_ppm"] = np.array([-1, 1])
I.VARPROparams["sampling_stepsize_fm"] = 0.1
I.VARPROparams["sampling_stepsize_r2s"] = 0.1
I.get_range_fm_Hz()
I.run_fieldmapping()
I.WFIparams["method"] = "sim_geo"
I.save_WFIparams()

methods = ["periodic1Ddrift", "drift"]
variants = ["1.0", "2.5", "5.0", "10.0", "15.0"]

for method in methods:
    for variant in variants:
        I = ImDataParamsBMRR("./data/sim/proc/" + "/20231205_165340_202_ImDataParamsBMRR_sim_" + method + "_" + variant + "_geo.h5")
        I.ImDataParams["voxelSize_mm"] = np.array([1.5, 1.5, 5.0])
        I.ImDataParams["signal"] = I.ImDataParams["signal"][:, :, :, 0, :]
        I.set_FatModel()
        I.set_VARPROparams()
        I.VARPROparams["range_fm_ppm"] = np.array([-1, 1])
        I.VARPROparams["sampling_stepsize_fm"] = 0.1
        I.VARPROparams["sampling_stepsize_r2s"] = 0.1
        I.get_range_fm_Hz()
        I.run_fieldmapping()
        I.WFIparams["method"] = "sim_" + method + "_" + variant + "_geo"
        I.save_WFIparams()