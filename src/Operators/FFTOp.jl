export cuFFTOp

mutable struct FFTOpImpl{T} <: FFTOp{T}
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
  iplan
  shift::Bool
  unitary::Bool
end

LinearOperators.storage_type(op::FFTOpImpl) = typeof(op.Mv5)

"""
  FFTOp(T::Type, shape::Tuple, shift=true, unitary=true)

returns an operator which performs an FFT on Arrays of type T

# Arguments:
* `T::Type`       - type of the array to transform
* `shape::Tuple`  - size of the array to transform
* (`shift=true`)  - if true, fftshifts are performed
* (`unitary=true`)  - if true, FFT is normalized such that it is unitary
"""
function cuFFTOp(T::Type, shape::NTuple{D,Int64}, shift::Bool=true; unitary::Bool=true, cuda::Bool=true) where D
  tmpVec = cuda ? CuArray{T}(undef,shape) : Array{Complex{real(T)}}(undef, shape)
  # @show typeof(tmpVec)
  # tmpVec = Array{Complex{real(T)}}(undef, shape)
  plan = plan_fft!(tmpVec)#; flags=FFTW.MEASURE)
  iplan = plan_bfft!(tmpVec)#; flags=FFTW.MEASURE)

  if unitary
    facF = T(1.0/sqrt(prod(shape)))
    facB = T(1.0/sqrt(prod(shape)))
  else
    facF = T(1.0)
    facB = T(1.0)
  end

  let shape_=shape, plan_=plan, iplan_=iplan, tmpVec_=tmpVec, facF_=facF, facB_=facB

  if shift
    return FFTOpImpl{T}(prod(shape), prod(shape), false, false
              , (res, x) -> fft_multiply_shift!(res, plan_, x, shape_, facF_, tmpVec_) 
              , nothing
              , (res, x) -> fft_multiply_shift!(res, iplan_, x, shape_, facB_, tmpVec_) 
              , 0, 0, 0, true, false, true, T[], T[]
              , plan
              , iplan
              , shift
              , unitary)
  else
    return FFTOpImpl{T}(prod(shape), prod(shape), false, false
            , (res, x) -> fft_multiply!(res, plan_, x, facF_, tmpVec_) 
            , nothing
            , (res, x) -> fft_multiply!(res, iplan_, x, facB_, tmpVec_)
            , 0, 0, 0, true, false, true, T[], T[]
            , plan
            , iplan
            , shift
            , unitary)
  end
  end
end

function fft_multiply!(res::AbstractVector{T}, plan::P, x::AbstractVector{Tr}, factor::T, tmpVec::AbstractArray{T,D}) where {T, Tr, P<:AbstractFFTs.Plan, D}
  tmpVec[:] .= x
  plan * tmpVec
  res .= factor .* vec(tmpVec)
end

function fft_multiply_shift!(res::AbstractVector{T}, plan::P, x::AbstractVector{Tr}, shape::NTuple{D}, factor::T, tmpVec::AbstractArray{T,D}) where {T, Tr, P<:AbstractFFTs.Plan, D}
  ifftshift!(tmpVec, reshape(x,shape))
  plan * tmpVec
  fftshift!(reshape(res,shape), tmpVec)
  res .*= factor
end

# function fft_multiply_shift!(res::AbstractVector{T}, plan::P, x::AbstractVector{Tr}, shape::NTuple{D}, factor::T, tmpVec::CuArray{T,D}) where {T, Tr, P<:AbstractFFTs.Plan, D}
#   ifftshift!(tmpVec, CuArray(reshape(x,shape)))
#   plan * tmpVec
#   tmpVec2 = similar(tmpVec)
#   fftshift!(reshape(tmpVec2,shape), tmpVec)
#   tmpVec2 .*= factor
#   copyto!(res, Array(tmpVec2))
# end

function Base.copy(S::LinearOperatorCollection.FFTOp)
  return cuFFTOp(eltype(S), size(S.plan), S.shift, unitary=S.unitary)
end
