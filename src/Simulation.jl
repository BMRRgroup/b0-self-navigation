export loadImgData, forwardSimulate, simulateRespiration, tra2Cor!, birdcageSensmaps!

function loadImgData(r::ReconParams{<:AbstractReconData, <:AbstractTrajectory}, filepath::String) 
    @debug("Load image data.")
    # Open HDF5 file
    fid = h5open(filepath, "r")
    signal = read(fid["ImDataParams"]["signal"])
    if ndims(signal) == 4
        signal = reshape(signal, size(signal, 1), 1, size(signal, 2), size(signal, 3), size(signal, 4))
    end
    signal = permutedims(signal, [5,4,3,1,2])
    signal = reshape(signal, size(signal, 1), size(signal, 2), size(signal, 3), size(signal, 4), size(signal, 5), 1, 1)
    data = ImgData(signal, nothing, nothing, nothing, nothing)
    append!(r.performedMethods, [nameof(var"#self#")])
    return ReconParams(r.filename, r.pathProc, r.scanParameters, r.reconParameters, 
            r.data, r.traj, r.performedMethods, data)
end

function forwardSimulate(r::ReconParams{<:AbstractKdata, <:NonCartesian3D, ImgData{T}}, traj::NonCartesian3D; multiCoil::Bool=false) where T<:AbstractFloat
    @debug("Forward stack-of-stars simulation.")
    if r.traj.name != :StackOfStars
        @error("Trajectory not implemented.")
    end
    img = r.imgData.signal
    reconSize = size(img)[1:3]
    numCoils = multiCoil ? size(r.reconParameters[:sensMaps], 4) : 1
    @show numCoils
    ksp = zeros(Complex{T}, size(traj.kdataNodes, 2), size(traj.kdataNodes, 3), size(img, 3), size(traj.kdataNodes, 4), 
        size(img, 5), numCoils, 1)
    for i in 1:size(img, 5)
        for j in 1:size(traj.kdataNodes, 4)
            if size(traj.kdataNodes, 5) == 1
                nodes = traj.kdataNodes[:,:,:,j,1]
            else
                nodes = traj.kdataNodes[:,:,:,j,i]
            end
            if multiCoil
                csm = r.reconParameters[:sensMaps]
                ft = DiagOp(repeat([StackOfStarsOp(reconSize, reshape(nodes,2,:), cuda=r.reconParameters[:cuda])], outer=numCoils)...)
                E = MRIOperators.CompositeOp(ft, MRIOperators.SensitivityOp(reshape(csm, prod(reconSize), numCoils)))
                ksp[:,:,:,j,i,:,1] .= reshape(E * reshape(img[:,:,:,j,i],:), size(ksp, 1), size(ksp, 2), size(img, 3), numCoils)
            else
                ft = StackOfStarsOp(reconSize, reshape(nodes,2,:), cuda=r.reconParameters[:cuda])
                ksp[:,:,:,j,i,1,1] .= reshape(ft * reshape(img[:,:,:,j,i],:), size(ksp, 1), size(ksp, 2), size(img, 3))
            end
        end
    end
    r.scanParameters[:encodingSize] = collect(size(img)[1:3])
    r.scanParameters[:KxOversampling] = 1.0
    r.scanParameters[:KyOversampling] = 1.0
    r.scanParameters[:KzOversampling] = 1.0
    data = KdataPreprocessed(ksp)
    return ReconParams(r.filename, r.pathProc, r.scanParameters, r.reconParameters, 
            data, traj, r.performedMethods, r.imgData)
end

function simulateRespiration(r::ReconParams{KdataPreprocessed{T}, <:NonCartesian3D}, cycle_duration_s::Float64; shotwise::Bool=true) where T<:AbstractFloat
    # Takes a kspaces of dimensions x,y,z,numTE,numMotionStates and combines the individual kspaces of each motion frame to correspond to
    # an MR acquisition of the XCAT phantom. Each spoke is taken from the kspace corresponding to a breathing motion frame of a simulated acquisition
    # inputs: kspaces (shape eg: (448, 449, 84, 4, 10) = (readout,numSpokes,slices,echoes,motionStates)), echoTimes TEs, cycle_duration (in s), starting_motion_frame
    # Returns combined kspace for all motion states
    @debug("Simulate respiration.")
    shot_duration = r.scanParameters[:shotDuration_ms] * 1e-3
    TR = r.scanParameters[:TR] * 1e-3
    ksp = r.data.kdata
    numStates = numDyn(r)
    numSlices = numKz(r)
    frame_duration = cycle_duration_s / numStates
    ksp_size = collect(size(ksp))
    ksp_size[5] = 1 # use dynamic dimension

    combined_ksp = zeros(Complex{T}, ksp_size...)
    motionStates = Vector(undef, numKy(r) * numSlices) 
    for i in 1:numKy(r) 
        for j in 1:numSlices
            if shotwise
                current_time_in_aquistion = (i-1) * shot_duration #+ (j-1) * TR 
            else
                current_time_in_aquistion = (i-1) * shot_duration + (j-1) * TR 
            end
            current_state = Int64(floor((current_time_in_aquistion / frame_duration) % numStates) + 1)
            motionStates[(i-1) * numSlices + j] = current_state
            # Take the spoke from current_state and fill appropriate place in combined_kspace
            combined_ksp[:,i,j,:,1,:,:] = ksp[:,i,j,:,current_state,:,:]
        end
    end
    r.reconParameters[:motionStatesSim] = motionStates
    r.reconParameters[:cycleDurationSim_s] = cycle_duration_s
    data = KdataPreprocessed(combined_ksp)
    return ReconParams(r.filename, r.pathProc, r.scanParameters, r.reconParameters, 
            data, r.traj, r.performedMethods, r.imgData)
end