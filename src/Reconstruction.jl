export iterativeRecon, iterativeRecon_perEchoDyn, initializeVariables

mutable struct initializedVariables{T}
    data::Array{Complex{T}, 7}
    traj::Union{Array{T, 5}, Nothing}
    reconSize::Tuple
    num_kx::Int
    num_chan::Int
    num_echoes::Int
    num_kz::Int
    num_ky::Int
    num_dyn::Int
    num_components::Union{Int, Nothing}
    num_dyn_virt_echoes::Int
    numMotionStates::Int
    numMotionStatesDeltaB0::Union{Int, Nothing}
    numMotionStatesRegulizer::Int
    bCurve::Vector{T}
end

function getRegularization(r::ReconParams{KdataPreprocessed{T}, <:AbstractTrajectory}, reconSize::Tuple, 
        numEchoesNumDyn::Int, numMotionStates::Int, numCoils::Int) where T<:AbstractFloat
    if r.reconParameters[:cudaSolver]
        opType = CuArray{Complex{T}, 1, CUDA.Mem.DeviceBuffer}
    else
        opType = Vector{Complex{T}}
    end

    reg = Vector{AbstractRegularization}()
    regTrafo = []
    if r.reconParameters[:iterativeReconParams][:Regularization][:L1Wavelet_spatial] > 0
        error("L1Wavelet_spatial not implemented.")
    end
    if r.reconParameters[:iterativeReconParams][:Regularization][:LLR] > 0
        error("LLR not implemented.")
        # push!(reg, LLRRegularization(T.(r.reconParameters[:iterativeReconParams][:Regularization][:LLR]), 
        #     shape=(reconSize[1], reconSize[2], reconSize[3], numEchoesNumDyn*numMotionStates*numCoils), 
        #     blockSize=(2,2,2,10)))
        # push!(regTrafo, opEye(T, prod((reconSize[1], reconSize[2], reconSize[3], 
        #     numEchoesNumDyn*numMotionStates*numCoils)), S=opType))
    end
    if r.reconParameters[:iterativeReconParams][:solver] == RegularizedLeastSquares.ADMM 
        if r.reconParameters[:iterativeReconParams][:Regularization][:TV_spatialTemporal] > 0 && numMotionStates > 1
            push!(reg, L1Regularization(T.(r.reconParameters[:iterativeReconParams][:Regularization][:TV_spatialTemporal])))
            push!(regTrafo, GradientOp(T; shape=(reconSize[1], reconSize[2], reconSize[3], 
                numEchoesNumDyn, numMotionStates, numCoils), dims=[1,2,3,5], S=opType))
        else
            if r.reconParameters[:iterativeReconParams][:Regularization][:TV_spatial] > 0
                push!(reg, L1Regularization(T.(r.reconParameters[:iterativeReconParams][:Regularization][:TV_spatial])))
                push!(regTrafo, GradientOp(T; shape=(reconSize[1], reconSize[2], reconSize[3], 
                    numEchoesNumDyn*numMotionStates*numCoils), dims=[1,2,3], S=opType))
            end
            if r.reconParameters[:iterativeReconParams][:Regularization][:TV_temporal] > 0 && numMotionStates > 1
                push!(reg, L1Regularization(T.(r.reconParameters[:iterativeReconParams][:Regularization][:TV_temporal])))
                push!(regTrafo, GradientOp(T; shape=(reconSize[1], reconSize[2], reconSize[3], 
                    numEchoesNumDyn, numMotionStates, numCoils), dims=5, S=opType))
            end
        end
    elseif r.reconParameters[:iterativeReconParams][:solver] == RegularizedLeastSquares.FISTA 
        if r.reconParameters[:iterativeReconParams][:Regularization][:TV_spatialTemporal] > 0 && numMotionStates > 1
            push!(reg, TVRegularization(T.(r.reconParameters[:iterativeReconParams][:Regularization][:TV_spatialTemporal]),
                shape=(reconSize[1], reconSize[2], reconSize[3], numEchoesNumDyn, numMotionStates, numCoils), dims=[1,2,3,5], S=opType))
        elseif r.reconParameters[:iterativeReconParams][:Regularization][:TV_spatial] > 0
            push!(reg, TVRegularization(T.(r.reconParameters[:iterativeReconParams][:Regularization][:TV_spatial]), 
                shape=(reconSize[1], reconSize[2], reconSize[3], numEchoesNumDyn*numMotionStates*numCoils), dims=[1,2,3], S=opType))
        end
    else
        error("Regularization for chosen solver not implemented.")
    end
    return reg, regTrafo
end

function initializeVariables(r::ReconParams{KdataPreprocessed{T}, <:AbstractTrajectory}; coilWise::Bool=false) where T<:AbstractFloat
    data = r.data.kdata
    params = r.reconParameters[:iterativeReconParams] # Recon paramters
    if typeof(r.traj) == Cartesian3D
        traj = nothing
        reconSize = size(r.data.kdata)[1:3]
    else
        traj = r.traj.kdataNodes
        reconSize = deepcopy(r.scanParameters[:encodingSize]) # Init with encodingSize
    end

    num_kx, num_chan, num_echoes, num_kz, num_ky, num_dyn = numKx(r), numChan(r), numEchoes(r), numKz(r), numKy(r), numDyn(r)

    ## get reconSize
    if coilWise
        if r.traj.name == :StackOfStars
            reconSize[1] = round(Int, reconSize[1] * r.scanParameters[:KxOversampling][1])
            reconSize[2] = round(Int, reconSize[2] * r.scanParameters[:KxOversampling][1])
            reconSize[3] = num_kz
        end
    else
        reconSize = size(r.reconParameters[:sensMaps])[1:3]
    end
    reconSize = Tuple(reconSize)

    num_components = nothing
    num_dyn_virt_echoes = num_dyn

    numMotionStates, bCurve = getBcurve(r)

    if params[:spatialDeltaB0]
        numMotionStatesDeltaB0 = numMotionStates
        numMotionStatesRegulizer = 1
    else
        numMotionStatesDeltaB0 = nothing
        numMotionStatesRegulizer = numMotionStates
    end

    return initializedVariables(data, traj, reconSize, num_kx, num_chan, num_echoes, num_kz, 
        num_ky, num_dyn, num_components, num_dyn_virt_echoes, numMotionStates, numMotionStatesDeltaB0, 
        numMotionStatesRegulizer, T.(bCurve))
end

function computeWeightsForSamplingDensity(r::ReconParams{KdataPreprocessed{T}, <:AbstractTrajectory}, maskMotion::Array, numContr::Int, trajTemp::Array, reconSize::Tuple, 
        num_kx::Int, num_ky::Int, num_kz::Int, num_chan::Int) where T<:AbstractFloat
    # Compute weights
    weights = similar(r.data.kdata, (num_kx*num_ky, numContr))
    for i = 1:numContr
        if :sdc in keys(r.reconParameters)
            weights[maskMotion[:,1,i,1],i] = sqrt.(reshape(r.reconParameters[:sdc], :))
        else
            if r.traj.name == :StackOfStars
                weights[maskMotion[:,1,i,1],i] = sqrt.(sdc(ReconBMRR.NFFTOp(Tuple(reconSize[1:2]), trajTemp[i], 
                    cuda=false).plan, iters=10)) # hardcoded for the first dynamic!
            else
                weights[maskMotion[:,1,i,1],i] = sqrt.(sdc(ReconBMRR.NFFTOp(reconSize, trajTemp[i], 
                    cuda=false).plan, iters=10)) # hardcoded for the first dynamic!
            end
        end
    end
    if r.traj.name == :StackOfStars
        weights .= weights .* Float32.(1/sqrt(num_kz))
    end
    weights = repeat(weights[:], num_chan*num_kz)
    weights = permutedims(reshape(weights, num_kx*num_ky, numContr, num_chan, num_kz), [1,4,2,3])[:]
    weightsMasked = weights[maskMotion[:]]
    return weightsMasked
end

function constructOperators(r::ReconParams{KdataPreprocessed{T}, <:AbstractTrajectory}, 
    weightsMasked::Array, numContr::Int, reconSize::Tuple, coilWise::Bool, num_chan::Int; 
    profiles::Array=[], trajTemp::Array=[], num_components=nothing, numMotionStatesDeltaB0=nothing, echoTime=nothing) where T<:AbstractFloat
    # if nan values occur, the kernelSize or oversamplingFactor of the nufft can be changed.

    cudaSolver = r.reconParameters[:cudaSolver] # Check if all operators are CUDA compatible and optimizer can run on the GPU
    if !r.reconParameters[:cuda] && cudaSolver
        error("CUDA solver is enabled but CUDA is not enabled in reconParameters.")
    end

    ## FFT operator
    if r.traj.name == :StackOfStars
        if r.reconParameters[:cuda] && cudaSolver
            @debug("StackOfStarsOp on GPU")
            ft = [CuStackOfStarsOp(reconSize, trajTemp[j], cuda=r.reconParameters[:cuda]) for j=1:numContr]
        else
            @debug("StackOfStarsOp")
            ft = [StackOfStarsOp(reconSize, trajTemp[j], cuda=r.reconParameters[:cuda]) for j=1:numContr]
        end
    else
        @debug("NFFTOp (not fully CUDA compatible)")
        ft = [ReconBMRR.NFFTOp(reconSize, trajTemp[j], cuda=r.reconParameters[:cuda]) for j=1:numContr]
    end

    if r.reconParameters[:cuda] && cudaSolver
        @debug("WeightingOp on GPU")
        cuWeightsMasked = CuArray(weightsMasked)
        weighting_op = LinearOperatorCollection.WeightingOp(cuWeightsMasked)
        E = CuCompositeOp(weighting_op, CuDiagOp(repeat(ft, outer=num_chan)...))
    else                    
        @debug("WeightingOp on CPU")
        # E = MRIOperators.CompositeOp(LinearOperatorCollection.WeightingOp(Complex{T}, weights=weightsMasked), DiagOp(ft...), isWeighting=true)
        E = MRIOperators.CompositeOp(LinearOperatorCollection.WeightingOp(weightsMasked),
            DiagOp(repeat(ft, outer=num_chan)...), isWeighting=true)
    end

    # Sensitivity operator
    if coilWise
        Efull = E
    else
        csm = rescaleSensMaps(r.reconParameters[:sensMaps])
        if r.reconParameters[:cuda] && cudaSolver
            @debug("SensitivityOp on GPU")
            Efull = CuCompositeOp(E, CuSensitivityOp2(reshape(csm, prod(reconSize), num_chan), numContr))
        else
            @debug("SensitivityOp on CPU")
            Efull = MRIOperators.CompositeOp(E, SensitivityOp2(reshape(csm, prod(reconSize), num_chan), numContr))
        end
    end

    # DeltaB0 operator
    if numMotionStatesDeltaB0 !== nothing
        if echoTime === nothing
            echoTime = r.scanParameters[:TE]
        end
        if r.reconParameters[:cuda] && cudaSolver
            @debug("DeltaB0Op on GPU")
            Efull = MRIOperators.CompositeOp(Efull, 
                CuDeltaB0Op(reshape(r.reconParameters[:deltaB0Field], prod(reconSize), numMotionStatesDeltaB0), 
                    echoTime, length(echoTime)))
        else
            @debug("DeltaB0Op on CPU")
            Efull = MRIOperators.CompositeOp(Efull, 
                DeltaB0Op(reshape(r.reconParameters[:deltaB0Field], prod(reconSize), numMotionStatesDeltaB0), 
                    echoTime, length(echoTime)))
        end
    end
    return Efull
end

function rescaleSensMaps(csm::Array{Complex{T}, 4}) where T<:AbstractFloat
    # Compute the sum of squares of magnitudes across coils
    sos = sqrt.(sum(abs.(csm).^2, dims=4))

    # Magnitude normalization
    normalized_array = similar(csm)
    normalized_array .= csm ./ (sos .+ eps())
    normalized_array[abs.(csm) .== 0] .= 0
    return normalized_array
end

function iterativeRecon(r::ReconParams{KdataPreprocessed{T}, <:AbstractTrajectory}; coilWise::Bool=false, init=nothing) where T<:AbstractFloat    
    @info("Iterative reconstruction.")
    params = r.reconParameters[:iterativeReconParams] # Recon paramters
    var = initializeVariables(r, coilWise=coilWise)
    
    numContr = var.num_echoes * var.num_dyn * var.numMotionStates 

    # Get regularization
    if coilWise
        reg, regTrafos = getRegularization(r, var.reconSize, var.num_echoes*var.num_dyn_virt_echoes, 
            var.numMotionStatesRegulizer, var.num_chan)
    else
        reg, regTrafos = getRegularization(r, var.reconSize, var.num_echoes*var.num_dyn_virt_echoes, 
            var.numMotionStatesRegulizer, 1)
    end

    @debug("Do motion sorting and get sampling mask.")
    data = reshape(permutedims(var.data, [1, 2, 5, 3, 4, 6, 7]), var.num_kx, :, var.num_kz, var.num_echoes, var.num_chan)
    if var.traj !== nothing
        traj = reshape(permutedims(var.traj, [1, 2, 3, 5, 4]), size(r.traj.kdataNodes,1), var.num_kx, :, var.num_echoes)
        trajTemp = zeros(T, size(r.traj.kdataNodes,1), var.num_kx, var.num_ky*var.num_dyn, var.num_echoes, var.numMotionStates)
    end
    dataTemp = zeros(Complex{T}, var.num_kx, var.num_ky*var.num_dyn, var.num_kz, var.num_echoes, var.numMotionStates, var.num_chan)
    maskMotion = zeros(Bool, var.num_kx, var.num_ky*var.num_dyn, var.num_echoes)
    for j in 1:var.numMotionStates
        idx = var.bCurve.==j
        dataTemp[:,idx,:,:,j,:] .= data[:,idx,:,:,:]
        if traj !== nothing
            trajTemp[:,:,idx,:,j] .= traj[:,:,idx,:]
        end
        maskMotion[:,idx,:] .= 1
    end
    # Transform back to arrays
    dataTemp = reorderArr(dataTemp, 
        [var.num_kx, var.num_ky, var.num_dyn, var.num_kz, var.num_echoes, var.numMotionStates, var.num_chan],
        [var.num_kx*var.num_ky, var.num_kz, numContr, var.num_chan], [1,2,4,5,3,6,7])
    maskMotion = repeat(maskMotion[:], var.num_kz*var.numMotionStates*var.num_chan)
    maskMotion = reorderArr(maskMotion, 
        [var.num_kx, var.num_ky, var.num_dyn, var.num_echoes, var.num_kz, var.numMotionStates, var.num_chan],
        [var.num_kx*var.num_ky, var.num_kz, numContr, var.num_chan], [1,2,5,4,3,6,7])

    dataTemp = dataTemp[maskMotion] 

    samplingMask = (dataTemp .== 0)

    trajTemp = reorderArr(trajTemp, 
        [size(r.traj.kdataNodes,1), var.num_kx, var.num_ky, var.num_dyn, var.num_echoes, var.numMotionStates], 
        [size(r.traj.kdataNodes,1), var.num_kx*var.num_ky, numContr], [1,2,3,5,4,6])
    trajTemp = [trajTemp[:,maskMotion[:,1,j,1] .== 1,j] for j=1:numContr]
    weightsMasked = computeWeightsForSamplingDensity(r, maskMotion, numContr, trajTemp, 
        var.reconSize, var.num_kx, var.num_ky, var.num_kz, var.num_chan)
    weightsMasked[samplingMask] .= 0
    dataTemp .= dataTemp .* weightsMasked
    Efull = constructOperators(r, weightsMasked, numContr, var.reconSize, coilWise, var.num_chan, 
        trajTemp=trajTemp, numMotionStatesDeltaB0=var.numMotionStatesDeltaB0)

    @debug("Create linear solver.")
    if init !== nothing
        startVector = reshape(init,:)
    else
        startVector = Vector{Complex{T}}(undef, 0)
    end

    solver = createLinearSolver(params[:solver], Efull; reg=reg, regTrafo=regTrafos, params...)

    if params[:verboseIteration]
        @warn("Verbose flag currently not supported.")
        # solver.verbose = true
    end
    if r.reconParameters[:cudaSolver] == true
        dataTemp_gpu = CuArray(dataTemp)
        startVector_gpu = CuArray(startVector)
    else
        dataTemp_gpu = dataTemp
        startVector_gpu = startVector
    end

    @debug("Perform optimization problem.")
    if init !== nothing
        @debug("Using initial guess.")
        img_gpu = solve!(solver, dataTemp_gpu, startVector=startVector_gpu) 
    else
        img_gpu = solve!(solver, dataTemp_gpu) 
    end
    img = Array(img_gpu)
    img = reshape(img, var.reconSize[1], var.reconSize[2], var.reconSize[3], var.num_echoes, 
        var.num_dyn_virt_echoes, var.numMotionStatesRegulizer, :)
    append!(r.performedMethods, [nameof(var"#self#")])

    # return data
    return_recon = ReconParams(r.filename, r.pathProc, r.scanParameters, r.reconParameters, 
    KdataRecon(dataTemp), r.traj, r.performedMethods, ImgData(img, nothing, nothing, nothing, nothing))
    return_recon.reconParameters[:Efull] = Efull
    return_recon.reconParameters[:reconSize] = var.reconSize
    return return_recon
end

function iterativeRecon_perEchoDyn(r::ReconParams{KdataPreprocessed{T}, <:AbstractTrajectory}; coilWise::Bool=false, verbose::Bool=true, iEchoes=nothing, iDyns=nothing) where T<:AbstractFloat
    @info("Iterative reconstruction.")
    params = r.reconParameters[:iterativeReconParams] # Recon paramters
    var = initializeVariables(r, coilWise=coilWise)
    
    if coilWise
        reg, regTrafos = getRegularization(r, var.reconSize, 1, var.numMotionStatesRegulizer, var.num_chan)
        img = zeros(Complex{T}, var.reconSize[1], var.reconSize[2], var.reconSize[3], var.num_echoes, var.num_dyn, 
            var.numMotionStatesRegulizer, var.num_chan)
    else
        reg, regTrafos = getRegularization(r, var.reconSize, 1, var.numMotionStatesRegulizer, 1)
        img = zeros(Complex{T}, var.reconSize[1], var.reconSize[2], var.reconSize[3], var.num_echoes, 
            var.num_dyn, var.numMotionStatesRegulizer, 1)
    end

    bCurve = reshape(var.bCurve, :, var.num_dyn)
    p = verboseProgress(var.num_echoes*var.num_dyn, "Perform reconstruction per echo and dynamic... ", verbose)

    if iEchoes === nothing
        iEchoes = 1:var.num_echoes
    end
    if iDyns === nothing
        iDyns = 1:var.num_dyn
    end
    combinations = [(i, j) for i in iEchoes for j in iDyns]

    for index in 1:length(combinations)
        let reg = reg, r = r
            i, j = combinations[index]
            dataTemp = zeros(Complex{T}, var.num_kx, var.num_ky, var.num_kz, var.numMotionStates, var.num_chan)
            trajTemp = zeros(T, size(r.traj.kdataNodes,1), var.num_kx, var.num_ky, var.numMotionStates)
            maskMotion = zeros(Bool, var.num_kx, var.num_ky)
            for k in 1:var.numMotionStates
                idx = bCurve[:,j] .== k
                dataTemp[:,idx,:,k,:] .= var.data[:,idx,:,i,j,:,:]
                trajTemp[:,:,idx,k] .= var.traj[:,:,idx,i,j]
                maskMotion[:,idx] .= 1
            end
            maskMotion = repeat(maskMotion[:], var.num_kz*var.numMotionStates*var.num_chan)
            dataTemp = reshape(dataTemp, var.num_kx*var.num_ky, var.num_kz, var.numMotionStates, var.num_chan)
            trajTemp = reshape(trajTemp, size(r.traj.kdataNodes,1), var.num_kx*var.num_ky, var.numMotionStates)
            maskMotion = reshape(maskMotion, var.num_kx*var.num_ky, var.num_kz, var.numMotionStates, var.num_chan)
            
            dataTemp = dataTemp[maskMotion] 
            trajTemp = [trajTemp[:,maskMotion[:,1,k,1] .== 1,k] for k=1:var.numMotionStates]
            samplingMask = (dataTemp .== 0)

            weightsMasked = computeWeightsForSamplingDensity(r, maskMotion, var.numMotionStates, trajTemp, 
                var.reconSize, var.num_kx, var.num_ky, var.num_kz, var.num_chan)
            weightsMasked[samplingMask] .= 0
            dataTemp .= dataTemp .* weightsMasked
            
            Efull = constructOperators(r, weightsMasked, var.numMotionStates, var.reconSize, coilWise, 
                var.num_chan, trajTemp=trajTemp, numMotionStatesDeltaB0=var.numMotionStatesDeltaB0)

            solver = createLinearSolver(params[:solver], Efull; reg=reg, regTrafo=regTrafos, params...)

            if params[:verboseIteration]
                @warn("Verbose flag currently not supported.")
                # solver.verbose = true
            end
            if r.reconParameters[:cudaSolver]
                dataTemp_gpu = CuArray(dataTemp)
                img_gpu = solve!(solver, dataTemp_gpu) 
                imgTmp = Array(img_gpu)
            else
                imgTmp = solve!(solver, dataTemp) 
            end
            img[:,:,:,i,j,:,:] = reshape(imgTmp, var.reconSize[1], var.reconSize[2], 
                var.reconSize[3], var.numMotionStatesRegulizer, :)
            verboseNext!(p)
        end
    end    
    append!(r.performedMethods, [nameof(var"#self#")])

    # return data
    return_recon = ReconParams(r.filename, r.pathProc, r.scanParameters, r.reconParameters, 
            r.data, r.traj, r.performedMethods, ImgData(img, nothing, nothing, nothing, nothing))    
    return_recon.reconParameters[:reconSize] = var.reconSize
    return return_recon
end