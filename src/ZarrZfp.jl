module ZarrZfp

# A self-contained, reusable zfp compressor for Zarr.jl. Wraps ZfpCompression.jl and plugs into
# Zarr's V2 compressor interface (`zcompress`/`zuncompress`/`getCompressor`/`JSON.lower` + the
# `compressortypes` registry).
#
# zfp is a compressor for 1–4D arrays of `Float32/Float64/Int32/Int64`. It compresses whole
# multidimensional blocks, so it must see the chunk WITH ITS SHAPE — Zarr's V2 pipeline passes the
# full `Array{T,N}` chunk to `zcompress`, exactly what zfp needs (no `reinterpret`/`vec` flattening,
# unlike the byte-stream codecs). Use whole-array (or at least ≥4-per-dim) chunks for a good ratio.
#
# On disk the codec is stored under the standard zfp Zarr codec id `"zfpy"`, carrying the zfp `mode`
# integer plus `tolerance`/`rate`/`precision` (unused ones set to -1). The compressed bytes are a plain
# zfp stream (with its full header), so any zfp-aware Zarr reader can decode them.

import Zarr
import JSON
using ZfpCompression: zfp_compress, zfp_decompress

export ZfpCompressor

# The libzfp `zfp_mode` enum integers (expert mode 1 is intentionally unsupported here). Values ARE
# the on-disk `mode` integers, so `Int(mode)` is the stored value. CamelCase names avoid clashing
# with the `precision`/`rate` keyword args and `Base.precision`.
@enum ZfpMode Rate=2 Precision=3 Accuracy=4 Reversible=5

"""
    ZfpCompressor(; tol, precision, rate)

A [Zarr](https://github.com/JuliaIO/Zarr.jl) `Compressor` backed by zfp (via ZfpCompression.jl).
Pass **exactly one** of the keywords to select the zfp mode, or none for lossless:

  * `tol::Real`         — *fixed-accuracy*: absolute error bound (max abs error ≈ `tol`).
  * `precision::Integer`— *fixed-precision*: number of uncompressed bit planes kept; gives an
    approximately constant, **scale-invariant** relative error (the natural choice for
    multi-decade / log-displayed fields).
  * `rate::Integer`     — *fixed-rate*: exact bits stored per value.
  * *(none)*            — *reversible*: lossless.

Stored under the standard zfp Zarr codec id `"zfpy"`, so the resulting store is a plain Zarr store
readable by any zfp-aware Zarr implementation.
"""
struct ZfpCompressor <: Zarr.Compressor
    mode::ZfpMode       # Accuracy | Precision | Rate | Reversible
    tol::Float64        # fixed-accuracy bound     (0 when unused)
    precision::Int      # fixed-precision bitplanes(0 when unused)
    rate::Int           # fixed-rate bits/value    (0 when unused)
end

function ZfpCompressor(; tol = nothing, precision = nothing, rate = nothing)
    n = count(!isnothing, (tol, precision, rate))
    n ≤ 1 || throw(ArgumentError("ZfpCompressor: pass at most one of `tol`, `precision`, `rate` (got $n)"))
    tol       !== nothing ? ZfpCompressor(Accuracy, tol, 0, 0) :
    precision !== nothing ? ZfpCompressor(Precision, 0.0, precision, 0) :
    rate      !== nothing ? ZfpCompressor(Rate, 0.0, 0, rate) :
                            ZfpCompressor(Reversible, 0.0, 0, 0)
end

# Mode → the ZfpCompression.jl `zfp_compress` keyword. Reversible passes nothing (its default).
_zfp_kw(z::ZfpCompressor) =
    z.mode === Accuracy  ? (; tol = z.tol) :
    z.mode === Precision ? (; precision = z.precision) :
    z.mode === Rate      ? (; rate = z.rate) :
                           NamedTuple()

# `a` is the full shaped chunk `Array{T,N}`; zfp writes a FULL header (type + dims), so the
# byte stream is self-describing for decompression.
Zarr.zcompress(a, z::ZfpCompressor) = zfp_compress(a; _zfp_kw(z)...)

# `a` is the compressed `Vector{UInt8}`; `zfp_decompress` reads the header and returns an
# `Array{T,N}` of the original element type & shape. Zarr's `zuncompress!` `copyto!`s it into the
# destination chunk (same shape & column-major order), so no reshape/reinterpret is needed.
Zarr.zuncompress(a, ::ZfpCompressor, T) = zfp_decompress(a)

function Zarr.getCompressor(::Type{ZfpCompressor}, d::Dict)
    m = d["mode"]
    m == Int(Rate)       ? ZfpCompressor(; rate = d["rate"]) :
    m == Int(Precision)  ? ZfpCompressor(; precision = d["precision"]) :
    m == Int(Accuracy)   ? ZfpCompressor(; tol = d["tolerance"]) :
    m == Int(Reversible) ? ZfpCompressor() :
        throw(ArgumentError("Unsupported zfp mode $m (zfp `expert` mode is not supported)"))
end

# `zfpy` codec config: every field present, unused params set to -1.
function JSON.lower(z::ZfpCompressor)
    Dict("id"        => "zfpy",
         "mode"      => Int(z.mode),
         "tolerance" => z.mode === Accuracy  ? z.tol       : -1,
         "rate"      => z.mode === Rate      ? z.rate      : -1,
         "precision" => z.mode === Precision ? z.precision : -1)
end

# Registering into Zarr's global registry MUST happen at load time (not top-level), otherwise the
# mutation is baked into the precompile image and lost on a fresh load.
function __init__()
    Zarr.compressortypes["zfpy"] = ZfpCompressor
end

end
