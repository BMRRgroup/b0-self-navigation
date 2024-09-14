export StackOfStarsOp, CuStackOfStarsOp
import Base.adjoint

mutable struct StackOfStarsOp{T} <: AbstractLinearOperator{T}
  nrow :: Int
  ncol :: Int
  symmetric :: Bool
  hermitian :: Bool
  prod! :: Function
  tprod! :: Nothing
  ctprod! :: Function
  nprod :: Int
  ntprod :: Int
  nctprod :: Int
  args5 :: Bool
  use_prod5! :: Bool
  allocated5 :: Bool
  Mv5 :: Vector{T}
  Mtu5 :: Vector{T}
  plan
  plan_z
  iplan_z
end

LinearOperators.storage_type(op::StackOfStarsOp) = typeof(op.Mv5)


mutable struct CuStackOfStarsOp{T, I <: Integer, F, Ft, Fct, S} <: AbstractLinearOperator{T}
  nrow::I
  ncol::I
  symmetric::Bool
  hermitian::Bool
  prod!::F
  tprod!::Ft
  ctprod!::Fct
  nprod::I
  ntprod::I
  nctprod::I
  args5::Bool
  use_prod5!::Bool # true for 5-args mul! and for composite operators created with operators that use the 3-args mul!
  Mv5::S
  Mtu5::S
  allocated5::Bool # true for 5-args mul!, false for 3-args mul! until the vectors are allocated
  plan :: Any  
  plan_z :: Any  
  iplan_z :: Any 
end

get_nargs(f) = first(methods(f)).nargs - 1

function CuStackOfStarsOp{T}(
  nrow::I,
  ncol::I,
  symmetric::Bool,
  hermitian::Bool,
  prod!::F,
  tprod!::Ft,
  ctprod!::Fct,
  nprod::I,
  ntprod::I,
  nctprod::I,
  plan,
  plan_z,
  iplan_z;
  S::DataType = Vector{T},
) where {T, I <: Integer, F, Ft, Fct}
  Mv5, Mtu5 = S(undef, 0), S(undef, 0)
  nargs = get_nargs(prod!)
  args5 = (nargs == 4)
  (args5 == false) || (nargs != 2) || throw(LinearOperatorException("Invalid number of arguments"))
  allocated5 = args5 ? true : false
  use_prod5! = args5 ? true : false
  return CuStackOfStarsOp{T, I, F, Ft, Fct, S}(
    nrow,
    ncol,
    symmetric,
    hermitian,
    prod!,
    tprod!,
    ctprod!,
    nprod,
    ntprod,
    nctprod,
    args5,
    use_prod5!,
    Mv5,
    Mtu5,
    allocated5,
    plan,
  plan_z,
  iplan_z
  )
end

LinearOperators.storage_type(op::CuStackOfStarsOp) = typeof(op.Mv5)



"""
    NFFTOp(shape::Tuple, tr::Trajectory; kargs...)
    NFFTOp(shape::Tuple, tr::AbstractMatrix; kargs...)

generates a `NFFTOp` which evaluates the MRI Fourier signal encoding operator using the NFFT.

# Arguments:
* `shape::NTuple{D,Int64}`  - size of image to encode/reconstruct
* `tr`                      - Either a `Trajectory` object, or a `ND x Nsamples` matrix for an ND-dimenensional (e.g. 2D or 3D) NFFT with `Nsamples` k-space samples
* (`nodes=nothing`)         - Array containg the trajectory nodes (redundant)
* (`kargs`)                 - additional keyword arguments
"""
function StackOfStarsOp(shape::Tuple, tr::AbstractMatrix{T}; cuda::Bool=true, oversamplingFactor=1.25, kernelSize=3, kargs...) where {T}
  if cuda
    plan = plan_nfft(CuArray, tr, shape[1:2], m=kernelSize, σ=oversamplingFactor)
    tmpVec = zeros(Complex{T}, size(tr,2),shape[3]) |> CuArray
    plan_z = plan_fft!(tmpVec, 2)
    iplan_z = plan_bfft!(tmpVec, 2)
    # Pre-allocate GPU arrays outside the function calls
    x_gpu = zeros(Complex{T}, prod(shape)) |> CuArray
    y_gpu = zeros(Complex{T}, size(tr,2)*shape[3]) |> CuArray
    return StackOfStarsOp{Complex{T}}(size(tr,2)*shape[3], prod(shape), false, false
                , (res,x) -> cuprodu!(res,plan,plan_z,x,shape,tmpVec,x_gpu,y_gpu)
                , nothing
                , (res,y) -> cuctprodu!(res,plan,iplan_z,y,shape,tmpVec,x_gpu,y_gpu)
                , 0, 0, 0, false, false, false, Complex{T}[], Complex{T}[]
                , plan, plan_z, iplan_z)
  else
    plan = plan_nfft(tr, shape[1:2], m=kernelSize, σ=oversamplingFactor, precompute=NFFT.TENSOR,
                    fftflags=FFTW.ESTIMATE, blocking=true)
    tmpVec = Array{Complex{real(T)}}(undef, (size(tr,2),shape[3]))
    plan_z = plan_fft!(tmpVec, 2; flags=FFTW.MEASURE)
    iplan_z = plan_bfft!(tmpVec, 2; flags=FFTW.MEASURE)
    return StackOfStarsOp{Complex{T}}(size(tr,2)*shape[3], prod(shape), false, false
                , (res,x) -> produ!(res,plan,plan_z,x,shape,tmpVec)
                , nothing
                , (res,y) -> ctprodu!(res,plan,iplan_z,y,shape,tmpVec)
                , 0, 0, 0, false, false, false, Complex{T}[], Complex{T}[]
                , plan, plan_z, iplan_z)
  end
end

function CuStackOfStarsOp(shape::Tuple, tr::AbstractMatrix{T}; oversamplingFactor=1.25, kernelSize=3, kargs...) where {T}
  plan = plan_nfft(CuArray, tr, shape[1:2], m=kernelSize, σ=oversamplingFactor)
  tmpVec = zeros(Complex{T}, size(tr,2),shape[3]) |> CuArray
  plan_z = plan_fft!(tmpVec, 2)
  iplan_z = plan_bfft!(tmpVec, 2)
  # println("plan: ", size(plan))
  # println("plan_z: ", size(plan_z))
  # println("iplan_z: ", size(iplan_z))
  # println("typeof: ", T)
  # x_gpu = zeros(Complex{T}, prod(shape)) |> CuArray
  # y_gpu = zeros(Complex{T}, size(tr,2)*shape[3]) |> CuArray
  return CuStackOfStarsOp{Complex{T}}(size(tr,2)*shape[3], prod(shape), false, false,
  (res,x) -> produ!(res, plan, plan_z, x, shape, tmpVec),
  # (res,x) -> testcuprodu!(res,plan,plan_z,x,shape,tmpVec,x_gpu,y_gpu),
  nothing,
  (res,y) -> ctprodu!(res, plan, iplan_z, y, shape, tmpVec),
  # (res,y) -> testcuctprodu!(res,plan,iplan_z,y,shape,tmpVec,x_gpu,y_gpu),
  0, 0, 0, plan, plan_z, iplan_z; S=CuArray{Complex{T}, 1, CUDA.Mem.DeviceBuffer}
  )
end
# plan, plan_z, ip; S=LinearOperators.storage_type(tmp)

function cuprodu!(y::AbstractVector, plan::AbstractNFFTs.AbstractNFFTPlan, plan_z::AbstractFFTs.Plan, 
    x::AbstractVector, shape::Tuple, tmpVec::AbstractArray, x_gpu::CuArray, y_gpu::CuArray) 
  copyto!(x_gpu, Array(x))
  fill!(y_gpu, zero(eltype(y)))
  produ!(y_gpu, plan, plan_z, x_gpu, shape, tmpVec)
  copyto!(y, Array(y_gpu))
end

function testcuprodu!(y::AbstractVector, plan::AbstractNFFTs.AbstractNFFTPlan, plan_z::AbstractFFTs.Plan, 
  x::AbstractVector, shape::Tuple, tmpVec::AbstractArray, x_gpu::CuArray, y_gpu::CuArray) 
  copyto!(x_gpu, CuArray(x))
  fill!(y_gpu, zero(eltype(y)))
  produ!(y_gpu, plan, plan_z, x_gpu, shape, tmpVec)
  copyto!(y, CuArray(y_gpu))
end

function cuctprodu!(x::AbstractVector, plan::AbstractNFFTs.AbstractNFFTPlan, plan_z::AbstractFFTs.Plan, 
    y::AbstractVector, shape::Tuple, tmpVec::AbstractArray, x_gpu::CuArray, y_gpu::CuArray)
  copyto!(y_gpu, Array(y))
  fill!(x_gpu, zero(eltype(x)))
  ctprodu!(x_gpu, plan, plan_z, y_gpu, shape, tmpVec)
  copyto!(x, Array(x_gpu))
end

function testcuctprodu!(x::AbstractVector, plan::AbstractNFFTs.AbstractNFFTPlan, plan_z::AbstractFFTs.Plan, 
  y::AbstractVector, shape::Tuple, tmpVec::AbstractArray, x_gpu::CuArray, y_gpu::CuArray)
  copyto!(y_gpu, y)
  fill!(x_gpu, zero(eltype(x)))
  ctprodu!(x_gpu, plan, plan_z, y_gpu, shape, tmpVec)
  copyto!(x, x_gpu)
end

function produ!(y::CuArray, plan::AbstractNFFTs.AbstractNFFTPlan, plan_z::AbstractFFTs.Plan, x::CuArray, shape::Tuple, tmpVec::CuArray)
  x = reshape(x, shape)  
  y = reshape(y, :, shape[3])
  ## NFFT
  for i=1:shape[3]
    mul!(view(y,:,i), plan, (view(x,:,:,i)))
  end
  fft_multiply_shift_sos!(plan_z, y, tmpVec) # FFT
  x = reshape(x, :)  
  y = reshape(y, :)
end

function ctprodu!(x::CuArray, plan::AbstractNFFTs.AbstractNFFTPlan, plan_z::AbstractFFTs.Plan, y::CuArray, shape::Tuple, tmpVec::CuArray)
  x = reshape(x, (shape))  
  y = reshape(y, :, shape[3])
  fft_multiply_shift_sos!(plan_z, y, tmpVec) # FFT
  ## NFFT
  for i=1:shape[3]
    mul!(view(x,:,:,i), adjoint(plan), (view(y,:,i)))
  end
  x = reshape(x, :)  
  y = reshape(y, :)
end

function fft_multiply_shift_sos!(plan::AbstractFFTs.Plan, y::CuArray, tmpVec::CuArray)
  ifftshift!(tmpVec, y)
  plan * tmpVec
  fftshift!(y, tmpVec)
  y *= 1/sqrt(size(tmpVec,2))
end

# function cuprodu!(y::AbstractVector, plan::AbstractNFFTs.AbstractNFFTPlan, plan_z::AbstractFFTs.Plan, x::AbstractVector, shape::Tuple, tmpVec::AbstractArray) 
#   x_gpu = CuArray(x)
#   y_gpu = CuArray(y)
#   x_gpu = reshape(x_gpu, shape)  
#   y_gpu = reshape(y_gpu, :, shape[3])  
#   cuprodu_kernel!(y_gpu, plan, x_gpu)
#   fft_multiply_shift!(y_gpu, plan_z, y_gpu, tmpVec) # FFT

#   copyto!(y, Array(reshape(y_gpu,:)))
# end

# function cuprodu_kernel!(y::CuArray, plan::AbstractNFFTs.AbstractNFFTPlan, x::CuArray) 
#   for i=1:size(x,3)
#     mul!(view(y,:,i), plan, view(x,:,:,i))
#   end
# end

# function cuctprodu!(x::AbstractVector, plan::AbstractNFFTs.AbstractNFFTPlan, plan_z::AbstractFFTs.Plan, y::AbstractVector, shape::Tuple, tmpVec::AbstractArray)
#   x_gpu = CuArray(x)
#   y_gpu = CuArray(y)
#   x_gpu = reshape(x_gpu, (shape))  
#   y_gpu = reshape(y_gpu, :, shape[3])  
#   fft_multiply_shift!(y_gpu, plan_z, y_gpu, tmpVec) # FFT
#   cuctprodu_kernel!(x_gpu, plan, y_gpu)

#   copyto!(x, Array(reshape(x_gpu,:)))
# end

# function cuctprodu_kernel!(x::CuArray, plan::AbstractNFFTs.AbstractNFFTPlan, y::CuArray)
#   for i=1:size(x,3)
#     mul!(view(x,:,:,i), adjoint(plan), view(y,:,i))
#   end
# end