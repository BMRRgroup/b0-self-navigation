export DeltaB0Op, CuDeltaB0Op

function prod_b0!(y::AbstractVector{T}, smaps, x::AbstractVector{T}, numVox, numChan, numContr, echoTimes) where T
  x_ = reshape(x,numVox,numContr)
  y_ = reshape(y,numVox,numContr,numChan)

  @assert size(smaps) == (size(y_,1), size(y_,3))

  @inbounds for i ∈ CartesianIndices(y_)
    y_[i] = x_[i[1],i[2]] * exp(+1im * 2 * pi * echoTimes[i[2]] * 1e-3 * smaps[i[1],i[3]])
  end
  return y
end

function ctprod_b0!(y::AbstractVector{T}, smapsC, x::AbstractVector{T}, numVox, numChan, numContr, echoTimes) where T
  x_ = reshape(x,numVox,numContr,numChan)
  y_ = reshape(y,numVox,numContr)

  @assert size(smapsC) == (size(x_,1), size(x_,3))

  y_ .= 0
  @inbounds for i ∈ CartesianIndices(x_)
    y_[i[1],i[2]] += x_[i] * exp(-1im * 2 * pi * echoTimes[i[2]] * 1e-3 * smapsC[i[1],i[3]]) #smapsC[i[1],i[3]]
  end
  return y
end

function DeltaB0Op(sensMaps, echoTimes, numContr=1)
    numVox, numChan = size(sensMaps)
    return LinearOperator{ComplexF32}(numVox*numContr*numChan, numVox*numContr, false, false,
                         (res,x) -> prod_b0!(res,sensMaps,x,numVox,numChan,numContr, echoTimes),
                         nothing,
                         (res,x) -> ctprod_b0!(res,sensMaps,x,numVox,numChan,numContr, echoTimes))
end

function cu_prod_b0!(y::CuArray{T}, smaps::CuArray{T}, x::CuArray{T}, numVox, numChan, numContr, echoTimes) where T
  threads_per_block = 256
  blocks_per_grid = (numVox + threads_per_block - 1) ÷ threads_per_block

  x_ = reshape(x,numVox,numContr)
  y_ = reshape(y,numVox,numContr,numChan)

  function prod_kernel!(y, smaps, x, numVox, numChan, numContr)
      voxel_idx = (blockIdx().x - 1) * blockDim().x  + (threadIdx().x)
      if voxel_idx <= numVox
          for contr_idx in 1:numContr
              for chan_idx in 1:numChan
                  y[voxel_idx, contr_idx, chan_idx] = x[voxel_idx, contr_idx] * exp(+1im * 2 * pi * echoTimes[contr_idx] * 1e-3 * smaps[voxel_idx, chan_idx])
              end
          end
      end
  
  end

  @cuda blocks=blocks_per_grid threads=threads_per_block prod_kernel!(y_, smaps, x_, numVox, numChan, numContr)
end

function cu_ctprod_b0!(y::CuArray{T}, smapsC::CuArray{T}, x::CuArray{T}, numVox, numChan, numContr=1) where T
  threads_per_block = 256
  blocks_per_grid = (numVox + threads_per_block - 1) ÷ threads_per_block

  x_ = reshape(x,numVox,numContr,numChan)
  y_ = reshape(y,numVox,numContr)

  function prod_kernel!(y, smapsC, x, numChan, numContr)
      voxel_idx = (blockIdx().x - 1) * blockDim().x  + (threadIdx().x)
      if voxel_idx <= numVox
          for contr_idx in 1:numContr
              y[voxel_idx, contr_idx] = zero(T)
              for chan_idx in 1:numChan
                  y[voxel_idx, contr_idx] += x[voxel_idx, contr_idx, chan_idx] * exp(-1im * 2 * pi * echoTimes[contr_idx] * 1e-3 * smapsC[voxel_idx, chan_idx])
              end
          end
      end
  
  end

  @cuda blocks=blocks_per_grid threads=threads_per_block prod_kernel!(y_, smapsC, x_, numChan, numContr)
end

function CuDeltaB0Op(sensMaps::AbstractMatrix{T}, echoTimes, numContr=1) where T
  numVox, numChan = size(sensMaps)
  cu_sensMaps = CuArray(sensMaps)
  tmp = CuArray(zeros(T, 1, 1))
  return LinearOperator{T}(numVox*numContr*numChan, numVox*numContr, false, false,
                       (res,x) -> cu_prod_b0!(res,cu_sensMaps,x,numVox,numChan,numContr,echoTimes),
                       nothing,
                       (res,x) -> cu_ctprod_b0!(res,cu_sensMaps,x,numVox,numChan,numContr,echoTimes),
                       S=LinearOperators.storage_type(tmp))
end