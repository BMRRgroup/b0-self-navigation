export ringingFilter!, artificalUndersampling!, changePrecision, noisePreWhitening!

function ringingFilter!(r::ReconParams{KdataPreprocessed{T}, <:AbstractTrajectory}; strength=0.25) where T<:AbstractFloat
    @debug("Apply ringing filter")
    dims = (r.traj.name == :StackOfStars) ? [1,3] : [1]
    r.data = KdataPreprocessed(hamming(r.data.kdata, strength=strength, dims=collect(dims)))
    append!(r.performedMethods, [nameof(var"#self#")])
end

function noisePreWhitening!(r::ReconParams{KdataPreprocessed{T}, <:AbstractTrajectory}) where T<:AbstractFloat
    if :Psi in keys(r.scanParameters)
        @debug("Perform noise pre-whitening.")
        data = r.data.kdata
        psi = r.scanParameters[:Psi]
        csm = r.reconParameters[:sensMaps]
        # Compute the Cholesky decomposition
        L_inv = inv(cholesky(psi).L)

        # Define the transformation function to be applied in-place
        function transform_fn!(slice)
            slice .= L_inv * slice
        end

        # Apply the transformation to the data using in-place mapslices with multi-threading
        @floop for s = eachslice(data, dims=(1, 2, 3, 4, 5, 7))
            transform_fn!(s)
        end

        # Similarly, apply the transformation to csm if needed
        @floop for s = eachslice(csm, dims=(1, 2, 3)) 
            transform_fn!(s)
        end

        r.data = KdataPreprocessed(data)
        r.reconParameters[:sensMaps] = csm
        append!(r.performedMethods, [nameof(var"#self#")])
    else
        error("Noise correlation matrix not found. No correction performed.")
    end
end

function artificalUndersampling!(r::ReconParams{KdataPreprocessed{T}, <:NonCartesian3D}) where T<:AbstractFloat
    @debug("Apply artifical undersampling.")
    removedKy = round(Int, numKy(r)*r.reconParameters[:artificalUndersampling])
    traj = r.traj.kdataNodes[:,:,1:end-removedKy,:,:]
    data = r.data.kdata[:,1:end-removedKy,:,:,:,:,:]
    if :motionStates in keys(r.reconParameters)
        bCurve = reshape(r.reconParameters[:motionStates], numKy(r), numDyn(r))
        bCurve = bCurve[1:end-removedKy, :]
        r.reconParameters[:motionStates] = reshape(bCurve,:)
    end
    r.data = KdataPreprocessed(data)
    r.traj = NonCartesian3D(traj, r.traj.name)
    append!(r.performedMethods, [nameof(var"#self#")])
end

function changePrecision(r::ReconParams{KdataPreprocessed{T}, J}) where {T<:AbstractFloat, J<:AbstractTrajectory}
    @debug("Change to double precision.")
    data = KdataPreprocessed(ComplexF64.(r.data.kdata))
    traj = J(Float64.(r.traj.kdataNodes), r.traj.name)
    if r.imgData !== nothing
        imgData = ImgData(ComplexF64.(r.imgData.signal), nothing, nothing, nothing, nothing)
    else
        imgData = nothing
    end
    r = ReconParams(r.filename, r.pathProc, r.scanParameters, r.reconParameters, 
        data, traj, r.performedMethods, imgData)
    if :sensMaps in keys(r.reconParameters)
        @debug("Precision of sensMaps changed to ComplexF64")
        r.reconParameters[:sensMaps] = ComplexF64.(r.reconParameters[:sensMaps])
    end
    append!(r.performedMethods, [nameof(var"#self#")])
    return r
end