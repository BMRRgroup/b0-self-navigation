export radialPhaseCorrection!

function radialPhaseCorrection!(r::ReconParams{KdataPreprocessed{T}, <:NonCartesian3D}; coilWise::Bool=true, saveShift::Bool=false, verbose::Bool=true) where T<:AbstractFloat
    # TODO: make parameters accesible from outside
    @debug("Perform radial spokes correction.")
    numKx = size(r.data.kdata, 1)
    # Change to profile order
    ky = r.reconParameters[:uniqueKy]

    data = deepcopy(r.data.kdata)
    mask = deepcopy(data .== 0)
    ifft_wshift!(data, 3)
    data[:,ky.+1,:,:,:,:] = data
    numSlices = size(data,3)
    meanCoils = true

    if coilWise
        coiloperation="coilwise"
    else
        coiloperation="coilcombined"
    end

    try
        shifts=h5read(r.filename[1:end-4]*"_shifts_spokealignment_"*coiloperation*".h5","shifts_spokealignment");
        r.reconParameters[:radialkSpaceShifts] = shifts
        println("Used shifts_spokealigned.h5 from directory " * dirname(r.filename))
    catch
        println("Calculate shifts...")

        # Reverse profile 2
        data_p1 = data[:,1:ceil(Int, numKy(r)/2),:,:,:,:]
        data_p2 = reverse(data[:,ceil(Int, numKy(r)/2):end,:,:,:,:], dims=1) # one spoke is used twice
        data_p2 = circshift(data_p2, (1,0,0,0,0,0)) # adjust trajectory offset in k-space by one sampling step size, numkx is from -N/2 to N/2-1 (aka -4, -3, -2, -1, 0, 1, 2, 3; to compare to reversed spoke we need shift by one step)

        # Go to image space
        tmpVec = zeros(eltype(data_p1), size(data_p1))
        iplan = plan_ifft!(tmpVec, 1)
        ReconBMRR.fft_multiply_shift!(iplan, data_p1, tmpVec)
        ReconBMRR.fft_multiply_shift!(iplan, data_p2, tmpVec)

        # per coil
        if coilWise
            padsize = 10 * numKx
            upsamplingFac = numKx/(2*padsize+numKx)
            num_chan = numChan(r)
            p = verboseProgress(num_chan, "Spoke alignment... ", verbose)
            # Zeropadding
            tmpVec = zeros(eltype(data_p1), 2*padsize+numKx, reduce(*, size(data_p1)[2:end-1]))
            plan = plan_fft!(tmpVec, 1)
            shifts = zeros(Int, size(tmpVec,2), num_chan)
            data_p1_pad = zeros(ComplexF32, 2*padsize+numKx, reduce(*, size(data_p1)[2:end-1]))
            data_p2_pad = zeros(ComplexF32, 2*padsize+numKx, reduce(*, size(data_p1)[2:end-1]))
            legs = -floor(Int, size(data_p1_pad,1)*0.05):floor(Int, size(data_p1_pad,1)*0.05)
            for j in 1:num_chan
                data_p1_pad .= 0
                data_p2_pad .= 0
                data_p1_pad[padsize+1:end-padsize,:] = reshape(data_p1[:,:,:,:,:,j], numKx, :)
                data_p2_pad[padsize+1:end-padsize,:] = reshape(data_p2[:,:,:,:,:,j], numKx, :)

                # FFT
                ReconBMRR.fft_multiply_shift!(plan, data_p1_pad, tmpVec)
                ReconBMRR.fft_multiply_shift!(plan, data_p2_pad, tmpVec)
                for i in eachindex(axes(data_p2_pad,2))#1:size(data_p1,2)
                    crosscorr = crosscor(abs.(data_p1_pad[:,i]), abs.(data_p2_pad[:,i]), legs)
                    shifts[i,j] = legs[argmax(crosscorr, dims=1)[1]]
                end
                verboseNext!(p)
            end
        else
            padsize = 10 * numKx
            upsamplingFac = numKx/(2*padsize+numKx)
            num_chan = 1
            data_p1 = reshape(data_p1, numKx, :, numChan(r))
            data_p2 = reshape(data_p2, numKx, :, numChan(r))
            p = verboseProgress(size(data_p1,2), "Spoke alignment... ", verbose)
            shifts = zeros(Int, size(data_p1,2), num_chan)

            tmpVec = zeros(eltype(data_p1), 2*padsize+numKx, numChan(r))
            plan = plan_fft!(tmpVec, 1)
            data_p1_pad = zeros(ComplexF32, 2*padsize+numKx, numChan(r))
            data_p2_pad = zeros(ComplexF32, 2*padsize+numKx, numChan(r))
            legs = -floor(Int, size(data_p1_pad,1)*0.05):floor(Int, size(data_p1_pad,1)*0.05)
            for i in eachindex(axes(data_p1,2))
                data_p1_pad .= 0
                data_p2_pad .= 0
                data_p1_pad[padsize+1:end-padsize,:] = data_p1[:,i,:]
                data_p2_pad[padsize+1:end-padsize,:] = data_p2[:,i,:]
                # FFT
                ReconBMRR.fft_multiply_shift!(plan, data_p1_pad, tmpVec)
                ReconBMRR.fft_multiply_shift!(plan, data_p2_pad, tmpVec)
                data_p1_sos = sum(abs.(data_p1_pad), dims=2)
                data_p2_sos = sum(abs.(data_p2_pad), dims=2)
                crosscorr = crosscor(data_p1_sos[:,1], data_p2_sos[:,1], legs) #plot crosscor maybe, would be more stable in image space
                shifts[i,1] = legs[argmax(crosscorr, dims=1)[1]]
                verboseNext!(p)
            end
        end

        # Mean over coils
        if meanCoils
            shifts = mean(reshape(shifts, ceil(Int, numKy(r)/2), :, num_chan), dims=3)[:,:,1] 
        else
            shifts = reshape(shifts, ceil(Int, numKy(r)/2), :)
        end
        shifts *= upsamplingFac
        r.reconParameters[:radialkSpaceShifts] = shifts

        shifts = vcat(shifts, shifts[2:end, :])
        if saveShift
            h5write(r.filename[1:end-4]*"_shifts_spokealignment_"*coiloperation*".h5", "shifts_spokealignment", shifts)
        end
    end


    # Correction
    data = reshape(data, numKx, numKy(r), :, numChan(r))
    ifft_wshift!(data,1)
    x = Array(-numKx/2:numKx/2-1) / (numKx/2)

    if meanCoils
        for i in eachindex(axes(shifts,2))  #dim 2 is echo*slices
            phase = exp.(1im*x*shifts[:,i]'*pi/2) #shifts[:,i]' is array of size(1, num_ky), phase has in the end a dimension of (num_kx, num_ky)
            for j in 1:numChan(r)
                data[:,:,i,j] .*= phase
            end
        end
    else
        shifts = reshape(shifts, numKy(r), :, num_chan)
        for i in eachindex(axes(shifts,2)) 
            for j in 1:numChan(r)
                phase = exp.(1im*x*shifts[:,i,j]'*pi/2)
                data[:,:,i,j] .*= phase
            end
        end        
    end

    fft_wshift!(data,1)
    data = reshape(data, numKx, numKy(r), numSlices, numEchoes(r), numDyn(r), numChan(r))

    # Change to temporal order
    data = data[:,ky.+1,:,:,:,:,:]
    # Radial stack of stars: ifft in z
    fft_wshift!(data, 3)
    data[mask] .= 0
    r.data = KdataPreprocessed(data)
    append!(r.performedMethods, [nameof(var"#self#")])
end
