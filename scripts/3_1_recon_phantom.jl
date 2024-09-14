using Pkg
Pkg.activate(".")
using Revise
using ReconBMRR
using JLD2, CodecZlib
using CUDA
CUDA.device!(0) ## select GPU

filename = "./data/phantom/scan1.jld2"
r2 = jldopen(filename)["r2"]
r2.pathProc = "./data/phantom/proc/"

noisePreWhitening!(r2)
radialPhaseCorrection!(r2)
ringingFilter!(r2, strength=0.35)
r2.reconParameters[:iterativeReconParams][:iterations] = 5
r2.reconParameters[:iterativeReconParams][:verboseIteration] = true
r2.reconParameters[:motionGating] = false

r3 = iterativeRecon_perEchoDyn(r2)
removeOversampling!(r3)
saveasImDataParams(r3, name="withoutB0Corr")

deltaB0Estimation!(r2, coilWise=true)
deltaB0Correction!(r2)
r2.reconParameters[:iterativeReconParams][:iterations] = 5
r2.reconParameters[:iterativeReconParams][:verboseIteration] = true
r2.reconParameters[:motionGating] = false
r3 = iterativeRecon_perEchoDyn(r2)
removeOversampling!(r3)
saveasImDataParams(r3, name="withB0Corr")