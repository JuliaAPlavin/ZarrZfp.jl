module ZarrZfp

import ZarrCore
using ZarrCore.Codecs.V3Codecs: V3Codecs, V3Codec
import JSON
using ZfpCompression: zfp_compress, zfp_decompress, zfp_decompress!

export ZfpCompressor, ZfpyCodec

@enum ZfpMode Rate=2 Precision=3 Accuracy=4 Reversible=5

"""
    ZfpCompressor(; tol, precision, rate)

A [Zarr](https://github.com/JuliaIO/Zarr.jl) `Compressor` backed by zfp (via ZfpCompression.jl).
Pass one of the keywords to select the zfp mode, or none for lossless:

  * `tol::Real`         — *fixed-accuracy*: absolute error bound (max abs error ≈ `tol`).
  * `precision::Integer`— *fixed-precision*: number of uncompressed bit planes kept; gives an approximately constant relative error
  * `rate::Real`        — *fixed-rate*: exact bits stored per value.
  * *(none)*            — *reversible*: lossless.
"""
struct ZfpCompressor <: ZarrCore.Compressor
    mode::ZfpMode
    tol::Float64
    precision::Int
    rate::Float64
end

function ZfpCompressor(; tol = nothing, precision = nothing, rate = nothing)
    n = count(!isnothing, (tol, precision, rate))
    n ≤ 1 || throw(ArgumentError("ZfpCompressor: pass at most one of `tol`, `precision`, `rate` (got $n)"))
    tol       !== nothing ? ZfpCompressor(Accuracy, tol, 0, 0) :
    precision !== nothing ? ZfpCompressor(Precision, 0, precision, 0) :
    rate      !== nothing ? ZfpCompressor(Rate, 0, 0, rate) :
                            ZfpCompressor(Reversible, 0, 0, 0)
end

_zfp_kw(z::ZfpCompressor) =
    z.mode === Accuracy  ? (; tol = z.tol) :
    z.mode === Precision ? (; precision = z.precision) :
    z.mode === Rate      ? (; rate = z.rate) :
                           NamedTuple()

# zfp pads every field axis to a full block, so a length-1 axis wastes a whole block.
# Squeeze length-1 axes before encoding; zfp records the shape, so decode stays unchanged.
_squeeze(a) = (dims = Tuple(findall(isone, size(a)));
               length(dims) < ndims(a) ? dropdims(a; dims) : a)
ZarrCore.zcompress(a, z::ZfpCompressor) = zfp_compress(_squeeze(a); _zfp_kw(z)...)

# zfp stores type and shape in the stream header.
ZarrCore.zuncompress(a, ::ZfpCompressor, T) = zfp_decompress(a)

# In-place read path: decode into Zarr's own dense output buffer, skipping the allocate-and-copy fallback
# (a non-dense fill-as-missing buffer isn't handled by zfp, so it takes that fallback). Squeeze length-1
# axes as `zcompress` does, so `data`'s shape matches the shape recorded in the header.
ZarrCore.zuncompress!(data::DenseArray, compressed, ::ZfpCompressor) = (zfp_decompress!(_squeeze(data), compressed); data)

function ZarrCore.getCompressor(::Type{ZfpCompressor}, d::Dict)
    m = d["mode"]
    m == Int(Rate)       ? ZfpCompressor(; rate = d["rate"]) :
    m == Int(Precision)  ? ZfpCompressor(; precision = d["precision"]) :
    m == Int(Accuracy)   ? ZfpCompressor(; tol = d["tolerance"]) :
    m == Int(Reversible) ? ZfpCompressor() :
        throw(ArgumentError("Unsupported zfp mode $m (zfp `expert` mode is not supported)"))
end

# The numcodecs `zfpy` parameter dict, shared by the v2 compressor and the v3 codec.
_zfpy_config(z::ZfpCompressor) =
    Dict("mode"      => Int(z.mode),
         "tolerance" => z.mode === Accuracy  ? z.tol       : -1,
         "rate"      => z.mode === Rate      ? z.rate      : -1,
         "precision" => z.mode === Precision ? z.precision : -1)

JSON.lower(z::ZfpCompressor) = merge(Dict("id" => "zfpy"), _zfpy_config(z))

"""
    ZfpyCodec(; tol, precision, rate)

The Zarr v3 array->bytes codec named `numcodecs.zfpy`, as in by zarr-python. The same zfp stream
(with header) and the same parameters as [`ZfpCompressor`](@ref).

```julia
zcreate(Float32, dims...; zarr_format = 3, codecs = (ZfpyCodec(tol = 1e-3),))
```
"""
struct ZfpyCodec <: V3Codec{:array, :bytes}
    compressor::ZfpCompressor
end
ZfpyCodec(; kw...) = ZfpyCodec(ZfpCompressor(; kw...))

V3Codecs.name(::ZfpyCodec) = "zfpy"

V3Codecs.codec_encode(c::ZfpyCodec, data::AbstractArray) = ZarrCore.zcompress(data, c.compressor)

# The header carries the encoded shape, which may or may not be squeezed,
# so decode from the header and restore the chunk shape afterwards.
function V3Codecs.codec_decode(c::ZfpyCodec, encoded::Vector{UInt8}, ::Type{T}, shape::NTuple{N,Int}; fill_value = nothing) where {T,N}
    decoded = ZarrCore.zuncompress(encoded, c.compressor, T)
    eltype(decoded) === T || throw(ArgumentError("zfp stream holds $(eltype(decoded)) values, but the array element type is $T"))
    reshape(decoded, shape)
end

JSON.lower(c::ZfpyCodec) = Dict("name" => "numcodecs.zfpy", "configuration" => _zfpy_config(c.compressor))

"""
    ZarrZfp.register!()

Register `ZfpCompressor` with ZarrCore under the Zarr v2 compressor name `"zfpy"`, and `ZfpyCodec`
under the Zarr v3 codec name `"zfpy"` (`"numcodecs.zfpy"` in metadata).
Runs automatically when the package loads unless the ZarrCore `RegisterAtInit` preference is disabled.
"""
function register!()
    ZarrCore.compressortypes["zfpy"] = ZfpCompressor
    V3Codecs.register_codec("zfpy", ZfpyCodec) do config, ctx
        ZfpyCodec(ZarrCore.getCompressor(ZfpCompressor, config))
    end
    nothing
end

# Registry mutation must happen at runtime, not during precompilation.
__init__() = ZarrCore.should_register_at_init() && register!()

end
