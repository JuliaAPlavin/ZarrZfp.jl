module ZarrZfp

import Zarr
import JSON
using ZfpCompression: zfp_compress, zfp_decompress

export ZfpCompressor

@enum ZfpMode Rate=2 Precision=3 Accuracy=4 Reversible=5

"""
    ZfpCompressor(; tol, precision, rate)

A [Zarr](https://github.com/JuliaIO/Zarr.jl) `Compressor` backed by zfp (via ZfpCompression.jl).
Pass one of the keywords to select the zfp mode, or none for lossless:

  * `tol::Real`         — *fixed-accuracy*: absolute error bound (max abs error ≈ `tol`).
  * `precision::Integer`— *fixed-precision*: number of uncompressed bit planes kept; gives an approximately constant relative error
  * `rate::Integer`     — *fixed-rate*: exact bits stored per value.
  * *(none)*            — *reversible*: lossless.
"""
struct ZfpCompressor <: Zarr.Compressor
    mode::ZfpMode
    tol::Float64
    precision::Int
    rate::Int
end

function ZfpCompressor(; tol = nothing, precision = nothing, rate = nothing)
    n = count(!isnothing, (tol, precision, rate))
    n ≤ 1 || throw(ArgumentError("ZfpCompressor: pass at most one of `tol`, `precision`, `rate` (got $n)"))
    tol       !== nothing ? ZfpCompressor(Accuracy, tol, 0, 0) :
    precision !== nothing ? ZfpCompressor(Precision, 0.0, precision, 0) :
    rate      !== nothing ? ZfpCompressor(Rate, 0.0, 0, rate) :
                            ZfpCompressor(Reversible, 0.0, 0, 0)
end

_zfp_kw(z::ZfpCompressor) =
    z.mode === Accuracy  ? (; tol = z.tol) :
    z.mode === Precision ? (; precision = z.precision) :
    z.mode === Rate      ? (; rate = z.rate) :
                           NamedTuple()

Zarr.zcompress(a, z::ZfpCompressor) = zfp_compress(a; _zfp_kw(z)...)

# zfp stores type and shape in the stream header.
Zarr.zuncompress(a, ::ZfpCompressor, T) = zfp_decompress(a)

function Zarr.getCompressor(::Type{ZfpCompressor}, d::Dict)
    m = d["mode"]
    m == Int(Rate)       ? ZfpCompressor(; rate = d["rate"]) :
    m == Int(Precision)  ? ZfpCompressor(; precision = d["precision"]) :
    m == Int(Accuracy)   ? ZfpCompressor(; tol = d["tolerance"]) :
    m == Int(Reversible) ? ZfpCompressor() :
        throw(ArgumentError("Unsupported zfp mode $m (zfp `expert` mode is not supported)"))
end

function JSON.lower(z::ZfpCompressor)
    Dict("id"        => "zfpy",
         "mode"      => Int(z.mode),
         "tolerance" => z.mode === Accuracy  ? z.tol       : -1,
         "rate"      => z.mode === Rate      ? z.rate      : -1,
         "precision" => z.mode === Precision ? z.precision : -1)
end

# Runtime registration survives precompilation.
function __init__()
    Zarr.compressortypes["zfpy"] = ZfpCompressor
end

end
