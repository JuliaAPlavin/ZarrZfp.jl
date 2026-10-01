using TestItems
using TestItemRunner
@run_package_tests


@testitem "round-trip through a Zarr store" begin
    using ZarrCore, ZarrZfp

    mk2d(T) = T[sin(x) * cos(y) + T(0.5) * x - T(0.3) * y for x in range(0, 2π, length = 64), y in range(0, 2π, length = 48)]
    mk3d(T) = T[sin(x) * cos(y) * sin(z) + T(0.2) * x for x in range(0, 2π, length = 32), y in range(0, 2π, length = 24), z in range(0, 2π, length = 16)]
    precision_bound(p, maxabs) = 2.0^(-(p - 7)) * maxabs
    rate16_bound(maxabs) = 5e-7 * maxabs

    arrays = [mk2d(Float64), mk3d(Float64), mk2d(Float32), mk3d(Float32)]
    for orig in arrays
        maxabs = maximum(abs, orig)
        cases = [
            (kw = (;),                bound = 0.0),
            (kw = (; tol = 1e-3),     bound = 1e-3),
            (kw = (; precision = 24), bound = precision_bound(24, maxabs)),
            (kw = (; rate = 16),      bound = rate16_bound(maxabs)),
        ]
        cidx = ntuple(_ -> Colon(), ndims(orig))

        for c in cases
            z = @inferred ZfpCompressor(; c.kw...)
            dir = mktempdir()
            arr = ZarrCore.zcreate(eltype(orig), size(orig)...; path = dir, chunks = size(orig), compressor = z)
            arr[cidx...] = orig
            back = ZarrCore.zopen(dir)[cidx...]

            @test eltype(back) == eltype(orig) && size(back) == size(orig)
            @test maximum(abs, back .- orig) ≤ c.bound
        end
    end
end


@testitem "compression shrinks size" begin
    using ZarrCore, ZarrZfp

    orig = Float64[sin(x) * cos(y) + 0.5x - 0.3y for x in range(0, 2π, length = 64), y in range(0, 2π, length = 48)]
    dir = mktempdir()
    z = ZarrCore.zcreate(Float64, size(orig)...; path = dir, chunks = size(orig), compressor = ZfpCompressor(rate = 16))
    z[:, :] = orig
    @test filesize(joinpath(dir, "0.0")) / sizeof(orig) ≈ 0.25 rtol = 0.01
end


@testitem "codec registration + reopen" begin
    using ZarrCore, ZarrZfp
    import JSON

    @test haskey(ZarrCore.compressortypes, "zfpy")
    @test ZarrCore.compressortypes["zfpy"] === ZfpCompressor

    @test ZarrCore.Codecs.V3Codecs.codec_parsers["zfpy"].return_type === ZfpyCodec

    delete!(ZarrCore.compressortypes, "zfpy")
    delete!(ZarrCore.Codecs.V3Codecs.codec_parsers, "zfpy")
    ZarrZfp.register!()
    @test ZarrCore.compressortypes["zfpy"] === ZfpCompressor
    @test ZarrCore.Codecs.V3Codecs.codec_parsers["zfpy"].return_type === ZfpyCodec

    for z in (ZfpCompressor(), ZfpCompressor(tol = 1e-3), ZfpCompressor(precision = 20), ZfpCompressor(rate = 16))
        @test ZarrCore.getCompressor(ZfpCompressor, JSON.lower(z)) == z
    end

    orig = Float64[sin(x) * cos(y) for x in range(0, 2π, length = 64), y in range(0, 2π, length = 48)]
    dir = mktempdir()
    z = ZarrCore.zcreate(Float64, size(orig)...; path = dir, chunks = size(orig), compressor = ZfpCompressor())
    z[:, :] = orig
    reopened = ZarrCore.zopen(dir)
    @test reopened.metadata.compressor == ZfpCompressor()
    @test reopened[:, :] == orig
end


@testitem "invalid configurations throw" begin
    using ZarrCore, ZarrZfp

    @test_throws ArgumentError ZfpCompressor(; tol = 1e-3, rate = 16)
    @test_throws ArgumentError ZfpCompressor(; tol = 1e-3, precision = 8, rate = 16)
    @test_throws ArgumentError ZarrCore.getCompressor(ZfpCompressor,
        Dict("mode" => 1, "tolerance" => -1, "rate" => -1, "precision" => -1))
end


@testitem "length-1 axes are squeezed before encoding" begin
    using ZarrCore, ZarrZfp

    # A length-1 axis carries no data, so it must encode identically to the same array
    # without it — regardless of where the length-1 axes sit or how many there are.
    base = Float32[sin(x) * cos(y) + 0.5f0 * x for x in range(0, 2π, length = 64), y in range(0, 2π, length = 48)]
    reshapes = [(64, 48, 1), (64, 1, 48), (1, 64, 48), (1, 64, 1, 48), (1, 64, 48, 1)]
    for z in (ZfpCompressor(), ZfpCompressor(tol = 1e-3), ZfpCompressor(precision = 20), ZfpCompressor(rate = 16))
        want = ZarrCore.zcompress(base, z)
        for shp in reshapes
            @test ZarrCore.zcompress(reshape(base, shp), z) == want
        end
    end
end


@testitem "round-trip through a store with a singleton chunk axis" begin
    using ZarrCore, ZarrZfp

    # A 3-D array chunked one plane at a time (chunks = (.., .., 1)) is the case the squeeze
    # targets. Non-multiple spatial sizes also exercise partial edge chunks.
    orig = Float32[sin(x) * cos(y) + 0.2f0 * z
                   for x in range(0, 2π, length = 70), y in range(0, 2π, length = 50), z in 1:5]
    maxabs = maximum(abs, orig)
    cases = [(z = ZfpCompressor(tol = 1e-3),     bound = 1e-3),
             (z = ZfpCompressor(precision = 24), bound = 2.0^(-(24 - 7)) * maxabs)]
    for c in cases
        dir = mktempdir()
        arr = ZarrCore.zcreate(Float32, size(orig)...; path = dir, chunks = (64, 48, 1), compressor = c.z)
        arr[:, :, :] = orig
        back = ZarrCore.zopen(dir)[:, :, :]
        @test size(back) == size(orig)
        @test maximum(abs, back .- orig) ≤ c.bound
    end

    # Degenerate all-length-1 chunks: nothing to squeeze, must still round-trip (lossless mode).
    small = Float32[x + 2y + 4z for x in 1:3, y in 1:3, z in 1:2]
    dir = mktempdir()
    arr = ZarrCore.zcreate(Float32, size(small)...; path = dir, chunks = (1, 1, 1), compressor = ZfpCompressor())
    arr[:, :, :] = small
    @test ZarrCore.zopen(dir)[:, :, :] == small
end


@testitem "zuncompress! decodes into the given buffer without an extra copy" begin
    using ZarrCore, ZarrZfp

    orig = Float32[sin(x) * cos(y) + 0.2f0 * z
                   for x in range(0, 2π, length = 40), y in range(0, 2π, length = 32), z in 1:8]

    # lossless: decode reproduces the original exactly, in dest's own memory, allocating no full buffer
    let z = ZfpCompressor(), comp = ZarrCore.zcompress(orig, z), dest = similar(orig)
        @test ZarrCore.zuncompress!(dest, comp, z) === dest
        @test dest == orig
        @test (@allocated ZarrCore.zuncompress!(dest, comp, z)) < sizeof(orig) ÷ 4
    end

    # lossy modes: in-place decode matches the allocating path bit-for-bit
    for z in (ZfpCompressor(tol = 1e-3), ZfpCompressor(precision = 24), ZfpCompressor(rate = 16))
        comp = ZarrCore.zcompress(orig, z)
        dest = similar(orig)
        @test ZarrCore.zuncompress!(dest, comp, z) == ZarrCore.zuncompress(comp, z, Float32)
    end

    # length-1 chunk axis: dest keeps the full shape while the header carries the squeezed shape
    let plane = reshape(orig[:, :, 1:1], size(orig, 1), size(orig, 2), 1),
        comp = ZarrCore.zcompress(plane, ZfpCompressor()), dest = similar(plane)
        @test ZarrCore.zuncompress!(dest, comp, ZfpCompressor()) === dest
        @test dest == plane
    end

    # a fill-as-missing chunk buffer is a non-dense SenMissArray, which zfp can't decode into: it must
    # take ZarrCore's allocate-and-copy fallback (our in-place method is restricted to `DenseArray`), not crash
    let dir = mktempdir()
        ZarrCore.zcreate(Float32, size(orig)...; path = dir, chunks = size(orig),
                     compressor = ZfpCompressor(), fill_value = NaN32)[:, :, :] = orig
        @test ZarrCore.zopen(dir; fill_as_missing = true)[:, :, :] == orig
    end
end


@testitem "_" begin
    import Aqua
    Aqua.test_all(ZarrZfp; ambiguities=false)
    Aqua.test_ambiguities(ZarrZfp)

    import CompatHelperLocal as CHL
    CHL.@check()
end


@testitem "fractional fixed rate" begin
    using ZarrCore, ZarrZfp
    import JSON

    # zfp rates are bits per value and need not be integral
    z = @inferred ZfpCompressor(rate = 12.5)
    @test ZarrCore.getCompressor(ZfpCompressor, JSON.lower(z)) == z
    @test JSON.lower(z)["rate"] == 12.5

    orig = Float32[sin(x) * cos(y) + 0.5f0 * x for x in range(0, 2π, length = 64), y in range(0, 2π, length = 48)]
    dir = mktempdir()
    arr = ZarrCore.zcreate(Float32, size(orig)...; path = dir, chunks = size(orig), compressor = z)
    arr[:, :] = orig
    @test filesize(joinpath(dir, "0.0")) / sizeof(orig) ≈ 12.5 / 32 rtol = 0.01
    @test maximum(abs, ZarrCore.zopen(dir)[:, :] .- orig) ≤ 1e-5
end


@testitem "zarr v3: round-trip, metadata, reopen" begin
    using ZarrCore, ZarrZfp
    import JSON

    orig = Float32[sin(x) * cos(y) * sin(z) + 0.2f0 * x for x in range(0, 2π, length = 32), y in range(0, 2π, length = 24), z in range(0, 2π, length = 16)]
    maxabs = maximum(abs, orig)
    cases = [
        (kw = (;),                bound = 0.0),
        (kw = (; tol = 1e-3),     bound = 1e-3),
        (kw = (; precision = 24), bound = 2.0^(-(24 - 7)) * maxabs),
        (kw = (; rate = 16),      bound = 5e-7 * maxabs),
    ]
    for c in cases
        codec = @inferred ZfpyCodec(; c.kw...)
        dir = mktempdir()
        arr = ZarrCore.zcreate(Float32, size(orig)...; path = dir, zarr_format = 3, chunks = size(orig), codecs = (codec,))
        arr[:, :, :] = orig

        # metadata is the numcodecs.zfpy codec with the same parameters as the v2 compressor
        meta = JSON.parsefile(joinpath(dir, "zarr.json"))
        @test meta["codecs"] == [Dict("name" => "numcodecs.zfpy", "configuration" => JSON.parse(JSON.json(ZarrZfp._zfpy_config(codec.compressor))))]

        # parsing inverts serialization, and the data comes back within the mode's bound
        reopened = ZarrCore.zopen(dir)
        @test reopened.metadata.pipeline.array_bytes == codec
        @test maximum(abs, reopened[:, :, :] .- orig) ≤ c.bound
    end
end


@testitem "zarr v3: singleton chunk axes" begin
    using ZarrCore, ZarrZfp

    orig = Float32[sin(x) * cos(y) + 0.2f0 * z
                   for x in range(0, 2π, length = 70), y in range(0, 2π, length = 50), z in 1:5]

    # length-1 chunk axes are squeezed as in v2: each plane chunk holds the 2-D encoding of that plane
    for kw in ((;), (; tol = 1e-3), (; precision = 24), (; rate = 16))
        dir = mktempdir()
        ZarrCore.zcreate(Float32, size(orig)...; path = dir, zarr_format = 3, chunks = (64, 48, 1), codecs = (ZfpyCodec(; kw...),))[:, :, :] = orig
        @test read(joinpath(dir, "c", "0", "0", "0")) == ZarrCore.zcompress(orig[1:64, 1:48, 1], ZfpCompressor(; kw...))
    end

    # all-length-1 chunks (nothing left to squeeze) and a fill_value/fill_as_missing read still work
    small = Float32[x + 2y + 4z for x in 1:3, y in 1:3, z in 1:2]
    dir = mktempdir()
    ZarrCore.zcreate(Float32, size(small)...; path = dir, zarr_format = 3, chunks = (1, 1, 1), codecs = (ZfpyCodec(),))[:, :, :] = small
    @test ZarrCore.zopen(dir)[:, :, :] == small
    dir = mktempdir()
    ZarrCore.zcreate(Float32, size(small)...; path = dir, zarr_format = 3, chunks = (2, 2, 2), codecs = (ZfpyCodec(),), fill_value = NaN32)[:, :, :] = small
    @test ZarrCore.zopen(dir; fill_as_missing = true)[:, :, :] == small
end


@testitem "zarr v3: reads arrays written by zarr-python" begin
    using ZarrCore, ZarrZfp
    import ZarrZstd  # zarr-python appends its default zstd codec after numcodecs.zfpy

    # zarr-python 3.4, serializer=ZFPY(mode=fixed_accuracy, tolerance=1e-3); its config omits the unused rate/precision keys
    z = ZarrCore.zopen(joinpath(@__DIR__, "fixtures", "zarr_python_v3_zfpy"))
    @test z.metadata.pipeline.array_bytes == ZfpyCodec(tol = 1e-3)
    @test z.metadata.pipeline.bytes_bytes == (ZarrZstd.ZstdV3Codec(0),)
    orig = Float32[sin(x) * cos(y) + 0.5f0 * x for x in range(0, 2π, length = 64), y in range(0, 2π, length = 48)]
    @test size(z) == size(orig)
    @test maximum(abs, z[:, :] .- orig) ≤ 1e-3

    # zarr-python does not squeeze, so the header shape (16, 12, 1) differs from what we would write for such chunks
    z = ZarrCore.zopen(joinpath(@__DIR__, "fixtures", "zarr_python_v3_zfpy_singleton"))
    @test z.metadata.chunks == (16, 12, 1)
    orig = Float32[sin(x) * cos(y) + 0.2f0 * z for x in range(0, 2π, length = 16), y in range(0, 2π, length = 12), z in 1:3]
    @test maximum(abs, z[:, :, :] .- orig) ≤ 1e-3
end
