export deltaB0Estimation!, deltaB0Correction!, motionStatesDeltaB0!, spatialDeltaB0estimation

function deltaB0Estimation!(r::ReconParams{KdataPreprocessed{T}, <:NonCartesian3D}; coilWise=true, doPlotMaps::Bool=true, verbose::Bool=true) where T<:AbstractFloat
    @debug("Estimate B0 variations.")
    data = getACregion(r, eddyCurrentCorrection=true, removeOversampling=false, kspaceOnly=false)
    if r.reconParameters[:LookLocker]
        data = reshape(permutedims(data, [3,2,1]), numKy(r)*numDyn(r), :, 1, numEchoes(r), 
            numChan(r))
    else
        data = reshape(permutedims(data, [2,3,1]), numKy(r)*numDyn(r), :, 1, numEchoes(r), 
            numChan(r))
    end
    numSlices = size(data,2)

    if numChan(r) > 1 && !coilWise
        csm = deepcopy(r.reconParameters[:sensMaps])
        fft_wshift!(csm, (1,2))
        centerX = floor(Int,size(csm,1)/2) + 1
        centerY = floor(Int,size(csm,2)/2) + 1
        csm = csm[centerX, centerY, :, :]

        csm = reshape(csm, 1, size(csm,1), 1, 1, size(csm,2))
        data = sum(csm .* data, dims=5)[:,:,:,:,1]
    elseif !coilWise
        data = data[:,:,:,:,1]
    end

    pygandalf = pyimport("hmrGC.dixon_imaging")
    th = 1

    fatModel = Dict{String, Any}()
    # TODO: hard coded fat model
    fatModel["freqs_ppm"] = [-3.8 , -3.4 , -3.1 , -2.68, -2.46, -1.95, -0.5 ,  0.49,  0.59]
    fatModel["relAmps"] = [0.08991009, 0.58341658, 0.05994006, 0.08491508, 0.05994006, 
                        0.01498501, 0.03996004, 0.00999001, 0.05694306]
    paramsGandalf = Dict{String, Any}()
    paramsGandalf["FatModel"] = fatModel 
    paramsGandalf["TE_s"] = r.scanParameters[:TE_s]
    paramsGandalf["centerFreq_Hz"] = r.scanParameters[:centerFreq_Hz]  # center frequency in Hz
    paramsGandalf["fieldStrength_T"] = 3  # field strength in T
    paramsGandalf["voxelSize_mm"] = [1.5, 1.5, 5.5]  # reconstruction voxel size in mm

    if !coilWise
        mip = maximum(abs.(data), dims=4)[:,:,:,1]
        mask = repeat(minimum(mip .> (th / 100 * percentile(reshape(mip,:),95)), dims=1),  outer=[size(mip,1), 1, 1])
        if numEchoes(r) > 2
            obj = pygandalf.MultiEcho(data, mask, paramsGandalf)
        end
        obj.range_fm_ppm = [-6,6]
        obj.sampling_stepsize_fm = 0.5
        obj.verbose = false
        obj.perform("single-res")
        phasormap = obj.fieldmap
        phasormap = reshape(phasormap, numKy(r)*numDyn(r), numSlices, 1, 1)
        mask = reshape(mask, size(mask,1), size(mask,2), size(mask,3), 1)
    else
        phasormap = zeros(Float64, size(data, 1), size(data, 2), size(data, 3), size(data, 5))
        water = zeros(ComplexF64, size(data, 1), size(data, 2), size(data, 3), size(data, 5))
        fat = zeros(ComplexF64, size(data, 1), size(data, 2), size(data, 3), size(data, 5))
        mip = maximum(abs.(data), dims=4)[:,:,:,1,:]
        mask = zeros(Bool, size(mip,1),size(mip,2),size(mip,3),size(mip,4))

        p = verboseProgress(size(mip,4), "Perform field-mapping per coil... ", verbose)
        # remove lines if there is no signal
        for i in eachindex(axes(mip,4))
            mask[:,:,:,i] = repeat(minimum(mip[:,:,:,i] .> (th / 100 * percentile(reshape(mip,:),95)), dims=1),  outer=[size(mip,1), 1, 1, 1])

            if sum(mask[:,:,:,i]) > 0
                if numEchoes(r) == 2
                    obj = pygandalf.DualEcho(data[:,:,:,1:2,i], mask[:,:,:,i], paramsGandalf)
                else
                    obj = pygandalf.MultiEcho(data[:,:,:,:,i], mask[:,:,:,i], paramsGandalf)
                end
                obj.range_fm_ppm = [-6,6]
                obj.sampling_stepsize_fm = 0.5
                obj.verbose = false
                obj.perform("single-res")
                phasormap[:,:,:,i] = obj.fieldmap
                water[:,:,:,i] = obj.images["water"]
                fat[:,:,:,i] = obj.images["fat"]
                verboseNext!(p)
            end
        end
    end

    # DEBUG
    # phasormap_to_return = deepcopy(phasormap)
    
    ## Look at fieldmap difference only
    if r.reconParameters[:LookLocker]
        phasormap = reshape(phasormap, numDyn(r), numKy(r), numKz(r), 1, size(phasormap, 4))
        phasormap = phasormap - repeat(median(phasormap, dims=2), outer=[1,size(phasormap,2),1,1,1])
        phasormap = reshape(phasormap, numDyn(r)*numKy(r), numKz(r), 1, size(phasormap, 5))
    else
        phasormap = phasormap - repeat(phasormap[1:1,:,:,:], outer=[size(phasormap,1),1,1,1])
    end

    coilCombination = "deltaB0smoothness"
    # coilCombination = "SNR_weighting"
    if coilCombination == "SNR_weighting"
        # ## Remove swaps
        mask_swaps = repeat(sum(abs.(phasormap[1:end-1,:,:,:].-phasormap[2:end,:,:,:]) .> 25, dims=1) .> 0, 
            outer=[size(phasormap,1),1,1,1])

        if coilWise
            mip[.!mask] .= 0
            mip[mask_swaps] .= 0
            @show size(mip)
            mip_median = repeat(median(mip, dims=1)[1,:,:,:], outer=[size(mip,1), 1, 1, 1])
            @show size(mip_median)
            weights = (mip ./ sum(mip, dims=4))[:,:,1,:]
            # weights = repeat(median((mip ./ sum(mip, dims=4))[:,:,1,:], dims=1), outer=[size(mip,1), 1, 1])
            weights[repeat(sum(mip, dims=4)[:,:,1,:], outer=[1,1,size(mip,4)]) .== 0] .= 0
            weights = reshape(weights, numKy(r)*numDyn(r), numSlices, numChan(r), 1)
            @assert maximum(sum(weights, dims=3)) - 1 < 1e-3
            phasormap = reshape(phasormap, numKy(r)*numDyn(r), numSlices, numChan(r), 1)
            if doPlotMaps
                numPlots = numChan(r)
                # numPlots = 2
                # numPlots = ceil(Int, numChan(r)/2)
                filename = joinpath(r.pathProc, basename(r.filename))
                splittedFilename = split(filename, ".")
                # plot([heatmap(transpose(phasormap[1:100,:,j,1]), c=:magma, clim=(-10,10)) for j in 1:numPlots]..., 
                #     layout=numPlots, colorbar=false)
                plot([plot(phasormap[1:100,1,j,1], c=:magma, clim=(-10,10)) for j in 1:numPlots]...) 
                savefig(join(splittedFilename[1:end-1], ".") * "_DeltaB0_channels.png")

                heatmap(weights[1,:,:,1], c=:viridis, colorbar=true, title="Coil weights")
                savefig(join(splittedFilename[1:end-1], ".") * "_DeltaB0_channels_weights.png")
            end
            phasormap = sum(weights[:,:,:,1] .* phasormap[:,:,:,1], dims=3) #mea
            phasormap[repeat(sum(weights, dims=3)[1:1,:,1] .== 0, outer=[size(weights,1),1,1])] .= NaN
        else
            phasormap = phasormap[:,:,:,1]
            mask_swaps = mask_swaps[:,:,:,1]
            phasormap[mask_swaps] .= NaN
            phasormap[mask .== 0] .= NaN
        end

        deltaB0 = nanmedian(phasormap, dims=2)[:,1,1]
        # phasormap[:, isnan.(phasormap[1,:])] .= deltaB0
        phasormap[:, isnan.(phasormap[1,:])] .= 0
        deltaB0_2d = phasormap[:,:,1] #mean(median(data[:,:,:,1,i],dims=2)[:,1,:],dims=2)[:,1]
    elseif coilCombination == "deltaB0smoothness"
        mask_signal = repeat(sum(abs.(phasormap), dims=1) .== 0, outer=[size(phasormap,1),1,1,1])
        phasormap[mask_signal] .= NaN
        smoothness = maximum(abs.(phasormap[1:end-1,:,:,:].-phasormap[2:end,:,:,:]), dims=1)
        deltaB0_2d = zeros(size(phasormap,1), size(phasormap,2))

        # Outlier rejection
        criterion = nanpctile(smoothness[1,:,1,:], 25, dim=2)
        q3 = nanpctile(criterion[:], 75)
        iqr = nanpctile(criterion[:], 75) - nanpctile(criterion[:], 25)
        global_criterion = q3 + 1.5*iqr
        for i=1:size(smoothness, 2)
            if criterion[i] < global_criterion
                phasormap[:,i,1,smoothness[1,i,1,:] .> global_criterion] .= NaN
                deltaB0_2d[:,i] .= nanmedian(phasormap[:,i,1,smoothness[1,i,1,:] .<= criterion[i]], dims=2)
            end
        end
        best_slice = nanargmin(criterion)
        deltaB0 = deltaB0_2d[:, best_slice]
    end

    ## Plot maps
    if doPlotMaps
        filename = joinpath(r.pathProc, basename(r.filename))
        splittedFilename = split(filename, ".")
        data = data[:,:,:,:,1]

        heatmap(transpose(angle.(data[:,:,1,1])), c=:magma, transpose=true, title="FH projection, phase")
        savefig(join(splittedFilename[1:end-1], ".") * "_DeltaB0_data_angle.png")

        heatmap(transpose(abs.(data[:,:,1,1])), c=:magma, transpose=true, title="FH projection")
        savefig(join(splittedFilename[1:end-1], ".") * "_DeltaB0_data.png")

        heatmap(transpose(deltaB0_2d)[:,:], c=:magma, transpose=true, title="DeltaB0")
        savefig(join(splittedFilename[1:end-1], ".") * "_DeltaB0.png")

        plot(deltaB0, title="Median DeltaB0")
        savefig(join(splittedFilename[1:end-1], ".") * "_DeltaB0_median.png")
    end
    r.reconParameters[:deltaB0] = deltaB0
    r.reconParameters[:deltaB0_2d] = deltaB0_2d
    append!(r.performedMethods, [nameof(var"#self#")])

    # DEBUG
    # return phasormap_to_return
end

function spatialDeltaB0estimation(r::ReconParams{<:AbstractKdata, <:AbstractTrajectory, ImgData{T}}, encodingSize; sensMapsSize=nothing, refState::Int=1) where T<:AbstractFloat
    pygandalf = pyimport("pygandalf.dixon_imaging")
    ndimage = pyimport("scipy.ndimage")

    data = r.imgData.signal

    th = 5
    fatModel = Dict{String, Any}()
    # TODO: hard coded fat model
    fatModel["freqs_ppm"] = [-3.8 , -3.4 , -3.1 , -2.68, -2.46, -1.95, -0.5 ,  0.49,  0.59]
    fatModel["relAmps"] = [0.08991009, 0.58341658, 0.05994006, 0.08491508, 0.05994006, 
                        0.01498501, 0.03996004, 0.00999001, 0.05694306]
    paramsGandalf = Dict{String, Any}()
    paramsGandalf["FatModel"] = fatModel 
    paramsGandalf["TE_s"] = vec(r.scanParameters[:TE])*1e-3 # echo times in s
    paramsGandalf["centerFreq_Hz"] = r.scanParameters[:centerFreq_Hz]  # center frequency in Hz
    paramsGandalf["fieldStrength_T"] = 3  # field strength in T
    paramsGandalf["voxelSize_mm"] = vec(r.scanParameters[:RecVoxelSize])  # reconstruction voxel size in mm

    phasormap = zeros(T, size(data, 1), size(data, 2), size(data, 3), size(data, 6))
    data = reshape(data, size(data, 1), size(data, 2), size(data, 3), size(data, 4), size(data, 6))
    mip = sqrt.(sum(abs.(data) .^ 2, dims=4)[:,:,:,1,:])
    mask = mip[:,:,:,1] .> th / 100 * maximum(mip[:,:,:,1])
    mask = ndimage.binary_fill_holes(mask)
    mask = correct_for_single_object(mask)
    for i in eachindex(axes(mip,4))
        @show i
        if size(data, 4) == 2
            obj = pygandalf.DualEcho(data[:,:,:,1:2,i], mask, paramsGandalf)
        else
            obj = pygandalf.MultiEcho(data[:,:,:,:,i], mask, paramsGandalf)
        end
        #obj.range_fm_ppm = [-5,5]
        obj.sampling_stepsize_fm = 0.5
        obj.verbose = false
        obj.perform("multi-res")
        phasormap[:,:,:,i] = obj.fieldmap
    end

    if sensMapsSize === nothing
        fac = encodingSize ./ r.scanParameters[:encodingSize]
        diff = zeros(Int(size(phasormap,1)*fac[1]), Int(size(phasormap,2)*fac[2]), Int(size(phasormap,3)*fac[3]), size(phasormap,4))
    else
        fac = collect(sensMapsSize[1:3] ./ size(phasormap)[1:3])
        diff = zeros(sensMapsSize[1], sensMapsSize[2], sensMapsSize[3], size(phasormap,4))
    end
    for i in eachindex(axes(diff,4))
        if i !== refState
            diff_state = phasormap[:,:,:,i] - phasormap[:,:,:,refState]
            median_filtered = ndimage.median_filter(diff_state, size=4)
            mask = (abs.(diff_state) .> 10) .&& (abs.(diff_state) .> 5*abs.(median_filtered))
            diff_state[mask] = median_filtered[mask]
            diff[:,:,:,i] = ndimage.zoom(diff_state, fac, order=0) #ndimage.gaussian_filter(ndimage.zoom(diff_state, fac, order=0), 1.0)
        end
    end
    return diff
end

function motionStatesDeltaB0!(r::ReconParams{KdataPreprocessed{T}, <:NonCartesian3D}; threshold_Hz = false) where T<:AbstractFloat
    fs = 1/(r.scanParameters[:shotDuration_ms]*1e-3)
    cutoff = 0.1
    wdo = 2.0 * cutoff / fs
    filth = digitalfilter(Highpass(wdo), Butterworth(20))
    bCurve = filtfilt(filth, r.reconParameters[:deltaB0])
    if !threshold_Hz
        params = r.reconParameters[:motionGatingParams]
        bCurveKz0 = bCurve
        # @show size(bCurveKz0)
        # @show quantile(bCurveKz0, 0.05)
        # @show quantile(bCurveKz0, 0.95)
        assignments, medoidsCenter = motionStatesClusteringStackOfStars(bCurveKz0, bCurveKz0, 
        numClusters=params[:numClusters], method=params[:method], numKz=numKz(r))
        r.reconParameters[:motionCurve] = bCurveKz0
        r.reconParameters[:motionStates] = assignments
        r.reconParameters[:motionStatesCenter] = medoidsCenter
        r.reconParameters[:motionMethod] = "SelfGating_"*String(params[:method])
        if params[:doPlotBcurve]
            plotBcurve(bCurveKz0, joinpath(r.pathProc, basename(r.filename)), assignments=assignments, numClusters=params[:numClusters])
        end
    else
        # gating based on threshold
        mask = abs.(bCurve) .> threshold_Hz
        motionStates = ones(Int, length(bCurve))
        motionStates[mask] .= 2
        r.reconParameters[:motionCurve] = bCurve
        r.reconParameters[:motionStates] = motionStates
        @info("Accepted data points: $(sum(mask .== 0)/length(bCurve)*100) %")
    end
end

function deltaB0Correction!(r::ReconParams{KdataPreprocessed{T}, <:NonCartesian3D}; coilWise::Bool=false) where T<:AbstractFloat
    @debug("Correct B0 fluctuations.")
    tes = r.scanParameters[:TE_s]
    numSlices = size(r.data.kdata,3)

    if r.reconParameters[:deltaB0Params][:corr2D] ## should be default
        @assert r.scanParameters[:HalfScanFactors][2] == 1.0
        deltaB0nav = r.reconParameters[:deltaB0_2d]
    else
        deltaB0nav = r.reconParameters[:deltaB0]
    end
    if r.reconParameters[:LookLocker] == true
        deltaB0nav = reshape(deltaB0nav, numDyn(r), :, size(deltaB0nav)[2:end]...)
        deltaB0nav = permutedims(deltaB0nav, [2, 1])
    else
        deltaB0nav = reshape(deltaB0nav, :, numDyn(r), size(deltaB0nav)[2:end]...)
    end

    for h in 1:numDyn(r) 
        kdata = r.data.kdata[:,:,:,:,h,:] 
        mask = deepcopy(kdata .== 0)
        ifft_wshift!(kdata, 3)
        for i in 1:numKy(r)
            for j in 1:numEchoes(r)
                if r.reconParameters[:deltaB0Params][:corr2D]
                    for k in 1:numSlices
                        if !coilWise
                            kdata[:,i,k,j,:] .*= exp(-1im*2*pi*tes[j]*deltaB0nav[i,h,k])
                        else
                            for l in 1:numChan(r)
                                kdata[:,i,k,j,:] .*= exp(-1im*2*pi*tes[j]*deltaB0nav[i,h,k,1,l])
                            end
                        end
                    end
                else
                    kdata[:,i,:,j,:] .*= exp(-1im*2*pi*tes[j]*deltaB0nav[i,h])
                end
            end
        end
        fft_wshift!(kdata, 3)
        kdata[mask] .= 0
        r.data.kdata[:,:,:,:,h,:] .= kdata
    end
end