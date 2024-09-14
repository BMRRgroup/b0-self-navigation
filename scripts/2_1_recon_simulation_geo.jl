using Pkg
Pkg.activate(".")
using Revise
using ReconBMRR
using JLD2, CodecZlib
using CUDA
CUDA.device!(0) ## select GPU

filename = "./data/sim/params.jld2"
r = jldopen(filename)["r"]
r3 = loadImgData(r, "./data/sim/geo_phantom_2_0mm.h5") 
r3.reconParameters[:iterativeReconParams][:iterations] = 5
r3.reconParameters[:iterativeReconParams][:verboseIteration] = true
r3.reconParameters[:motionGating] = false

## Ground truth
r4 = forwardSimulate(r3, r3.traj)
r4.pathProc = "./data/sim/proc/"
data = ReconBMRR.KdataPreprocessed(r4.data.kdata[:,:,:,:,1:1,:,:])
r4 = ReconParams(r4.filename, r4.pathProc, r4.scanParameters, r4.reconParameters, 
    data, r4.traj, r4.performedMethods, r4.imgData)
r5 = iterativeRecon(r4, coilWise=true)
saveasImDataParams(r5, name="sim_geo")

## Drift
for maxB0drift in [1, 2.5, 5, 10, 15]
    r4 = forwardSimulate(r3, r3.traj)
    r4.pathProc = "./data/sim/proc/"
    r4.reconParameters[:deltaB0] = - collect(0:maxB0drift/(ReconBMRR.numKy(r4)-1):maxB0drift)
    r4.reconParameters[:deltaB0Params][:corr2D] = false
    deltaB0Correction!(r4)
    r4.reconParameters[:deltaB0Params][:corr2D] = true
    r5 = iterativeRecon(r4, coilWise=true)
    saveasImDataParams(r5, name="sim_drift_" * string(maxB0drift) *"_geo")
end

## 1D periodic B0 fluctuations
for amp in [1, 2.5, 5, 10, 15]
    r4 = forwardSimulate(r3, r3.traj)
    # select only one motion state
    data = ReconBMRR.KdataPreprocessed(r4.data.kdata[:,:,:,:,1:1,:,:])
    r4 = ReconParams(r4.filename, r4.pathProc, r4.scanParameters, r4.reconParameters, data, r4.traj, r4.performedMethods, r4.imgData)
    r4.pathProc = "./data/sim/proc/"
    period = 1/5
    shotDur = r.scanParameters[:shotDuration_ms] * 1e-3
    t = collect(0:shotDur:shotDur*ReconBMRR.numKy(r4))
    maxB0drift = 10
    r4.reconParameters[:deltaB0] = - amp * sin.(2*pi*period*t) - collect(0:maxB0drift/(ReconBMRR.numKy(r4)):maxB0drift)
    r4.reconParameters[:deltaB0Params][:corr2D] = false
    deltaB0Correction!(r4)
    r4.reconParameters[:deltaB0Params][:corr2D] = true
    r5 = iterativeRecon(r4, coilWise=true)
    saveasImDataParams(r5, name="sim_periodic1Ddrift_" * string(amp) *"_geo")
end