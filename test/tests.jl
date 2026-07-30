using TestItems

# Round-trip whole arrays through a real on-disk Zarr store (single whole-array chunk) using
# `ZfpCompressor` as the chunk compressor — across shapes (2D/3D) and element types (Float32/Float64),
# for every zfp mode. Bounds are derived from zfp semantics (never by recompressing here) and are
# expressed uniformly as `maximum(abs, back .- orig) ≤ bound` so a failure prints both operands:
#   * reversible : bound 0     — lossless, exact.
#   * accuracy   : bound = tol — zfp's fixed-accuracy guarantee is an absolute max-error bound.
#   * precision  : zfp keeps ~`p` bit planes ⇒ relative error ≈ 2^-p up to a block-transform slack
#                  measured at ~2^5.7; we bound by 2^-(p-7) (slack < 2^7), ~2.5× above the worst case.
#   * rate       : 16 bits/value of correlated data ⇒ relative error ~1.8e-7; bound 5e-7 (~2.7× margin).
@testitem "round-trip through a Zarr store" begin
    using Zarr, ZarrZfp

    mk2d(T) = T[sin(x) * cos(y) + T(0.5) * x - T(0.3) * y for x in range(0, 2π, length = 64), y in range(0, 2π, length = 48)]
    mk3d(T) = T[sin(x) * cos(y) * sin(z) + T(0.2) * x for x in range(0, 2π, length = 32), y in range(0, 2π, length = 24), z in range(0, 2π, length = 16)]
    arrays = [mk2d(Float64), mk3d(Float64), mk2d(Float32), mk3d(Float32)]

    for orig in arrays
        maxabs = maximum(abs, orig)
        cases = [
            (kw = (;),                bound = 0.0),                         # reversible: exact
            (kw = (; tol = 1e-3),     bound = 1e-3),                        # fixed-accuracy
            (kw = (; precision = 24), bound = 2.0^(-(24 - 7)) * maxabs),    # fixed-precision
            (kw = (; rate = 16),      bound = 5e-7 * maxabs),              # fixed-rate
        ]
        cidx = ntuple(_ -> Colon(), ndims(orig))
        for c in cases
            z = @inferred ZfpCompressor(; c.kw...)
            dir = mktempdir()
            arr = Zarr.zcreate(eltype(orig), size(orig)...; path = dir, chunks = size(orig), compressor = z)
            arr[cidx...] = orig
            back = Zarr.zopen(dir)[cidx...]
            @test eltype(back) == eltype(orig) && size(back) == size(orig)
            @test maximum(abs, back .- orig) ≤ c.bound
        end
    end
end

# Fixed-rate compression is deterministic: rate=16 bits stores 2 of every 8 bytes of a Float64 value,
# so the single chunk file is 1/4 the raw size (plus a negligible zfp header).
@testitem "compression shrinks size" begin
    using Zarr, ZarrZfp

    orig = Float64[sin(x) * cos(y) + 0.5x - 0.3y for x in range(0, 2π, length = 64), y in range(0, 2π, length = 48)]
    dir = mktempdir()
    z = Zarr.zcreate(Float64, size(orig)...; path = dir, chunks = size(orig), compressor = ZfpCompressor(rate = 16))
    z[:, :] = orig
    @test filesize(joinpath(dir, "0.0")) / sizeof(orig) ≈ 0.25 rtol = 0.01   # 16/64 bits + header
end

# The codec registers into Zarr's global registry, survives a fresh store reopen (via the on-disk
# `"zfpy"` id → `getCompressor`), and its config round-trips losslessly through `JSON.lower`.
@testitem "codec registration + reopen" begin
    using Zarr, ZarrZfp
    import JSON

    @test haskey(Zarr.compressortypes, "zfpy")
    @test Zarr.compressortypes["zfpy"] === ZfpCompressor

    # JSON.lower (String-keyed) → getCompressor reconstructs the exact same compressor, for every mode.
    for z in (ZfpCompressor(), ZfpCompressor(tol = 1e-3), ZfpCompressor(precision = 20), ZfpCompressor(rate = 16))
        @test Zarr.getCompressor(ZfpCompressor, JSON.lower(z)) == z
    end

    orig = Float64[sin(x) * cos(y) for x in range(0, 2π, length = 64), y in range(0, 2π, length = 48)]
    dir = mktempdir()
    z = Zarr.zcreate(Float64, size(orig)...; path = dir, chunks = size(orig), compressor = ZfpCompressor())
    z[:, :] = orig
    reopened = Zarr.zopen(dir)                                  # reads .zarray → resolves "zfpy" → getCompressor
    @test reopened.metadata.compressor == ZfpCompressor()
    @test reopened[:, :] == orig                               # decodes via the registered codec
end

# Invalid configurations must fail loud.
@testitem "invalid configurations throw" begin
    using Zarr, ZarrZfp

    @test_throws ArgumentError ZfpCompressor(; tol = 1e-3, rate = 16)                 # more than one mode
    @test_throws ArgumentError ZfpCompressor(; tol = 1e-3, precision = 8, rate = 16)  # all three
    # zfp `expert` mode (1) is intentionally unsupported by getCompressor.
    @test_throws ArgumentError Zarr.getCompressor(ZfpCompressor,
        Dict("mode" => 1, "tolerance" => -1, "rate" => -1, "precision" => -1))
end
