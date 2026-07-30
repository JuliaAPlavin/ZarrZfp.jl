using TestItems
using TestItemRunner
@run_package_tests


@testitem "round-trip through a Zarr store" begin
    using Zarr, ZarrZfp

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
            arr = Zarr.zcreate(eltype(orig), size(orig)...; path = dir, chunks = size(orig), compressor = z)
            arr[cidx...] = orig
            back = Zarr.zopen(dir)[cidx...]

            @test eltype(back) == eltype(orig) && size(back) == size(orig)
            @test maximum(abs, back .- orig) ≤ c.bound
        end
    end
end


@testitem "compression shrinks size" begin
    using Zarr, ZarrZfp

    orig = Float64[sin(x) * cos(y) + 0.5x - 0.3y for x in range(0, 2π, length = 64), y in range(0, 2π, length = 48)]
    dir = mktempdir()
    z = Zarr.zcreate(Float64, size(orig)...; path = dir, chunks = size(orig), compressor = ZfpCompressor(rate = 16))
    z[:, :] = orig
    @test filesize(joinpath(dir, "0.0")) / sizeof(orig) ≈ 0.25 rtol = 0.01
end


@testitem "codec registration + reopen" begin
    using Zarr, ZarrZfp
    import JSON

    @test haskey(Zarr.compressortypes, "zfpy")
    @test Zarr.compressortypes["zfpy"] === ZfpCompressor

    for z in (ZfpCompressor(), ZfpCompressor(tol = 1e-3), ZfpCompressor(precision = 20), ZfpCompressor(rate = 16))
        @test Zarr.getCompressor(ZfpCompressor, JSON.lower(z)) == z
    end

    orig = Float64[sin(x) * cos(y) for x in range(0, 2π, length = 64), y in range(0, 2π, length = 48)]
    dir = mktempdir()
    z = Zarr.zcreate(Float64, size(orig)...; path = dir, chunks = size(orig), compressor = ZfpCompressor())
    z[:, :] = orig
    reopened = Zarr.zopen(dir)
    @test reopened.metadata.compressor == ZfpCompressor()
    @test reopened[:, :] == orig
end


@testitem "invalid configurations throw" begin
    using Zarr, ZarrZfp

    @test_throws ArgumentError ZfpCompressor(; tol = 1e-3, rate = 16)
    @test_throws ArgumentError ZfpCompressor(; tol = 1e-3, precision = 8, rate = 16)
    @test_throws ArgumentError Zarr.getCompressor(ZfpCompressor,
        Dict("mode" => 1, "tolerance" => -1, "rate" => -1, "precision" => -1))
end


@testitem "length-1 axes are squeezed before encoding" begin
    using Zarr, ZarrZfp

    # A length-1 axis carries no data, so it must encode identically to the same array
    # without it — regardless of where the length-1 axes sit or how many there are.
    base = Float32[sin(x) * cos(y) + 0.5f0 * x for x in range(0, 2π, length = 64), y in range(0, 2π, length = 48)]
    reshapes = [(64, 48, 1), (64, 1, 48), (1, 64, 48), (1, 64, 1, 48), (1, 64, 48, 1)]
    for z in (ZfpCompressor(), ZfpCompressor(tol = 1e-3), ZfpCompressor(precision = 20), ZfpCompressor(rate = 16))
        want = Zarr.zcompress(base, z)
        for shp in reshapes
            @test Zarr.zcompress(reshape(base, shp), z) == want
        end
    end
end


@testitem "round-trip through a store with a singleton chunk axis" begin
    using Zarr, ZarrZfp

    # A 3-D array chunked one plane at a time (chunks = (.., .., 1)) is the case the squeeze
    # targets. Non-multiple spatial sizes also exercise partial edge chunks.
    orig = Float32[sin(x) * cos(y) + 0.2f0 * z
                   for x in range(0, 2π, length = 70), y in range(0, 2π, length = 50), z in 1:5]
    maxabs = maximum(abs, orig)
    cases = [(z = ZfpCompressor(tol = 1e-3),     bound = 1e-3),
             (z = ZfpCompressor(precision = 24), bound = 2.0^(-(24 - 7)) * maxabs)]
    for c in cases
        dir = mktempdir()
        arr = Zarr.zcreate(Float32, size(orig)...; path = dir, chunks = (64, 48, 1), compressor = c.z)
        arr[:, :, :] = orig
        back = Zarr.zopen(dir)[:, :, :]
        @test size(back) == size(orig)
        @test maximum(abs, back .- orig) ≤ c.bound
    end

    # Degenerate all-length-1 chunks: nothing to squeeze, must still round-trip (lossless mode).
    small = Float32[x + 2y + 4z for x in 1:3, y in 1:3, z in 1:2]
    dir = mktempdir()
    arr = Zarr.zcreate(Float32, size(small)...; path = dir, chunks = (1, 1, 1), compressor = ZfpCompressor())
    arr[:, :, :] = small
    @test Zarr.zopen(dir)[:, :, :] == small
end


@testitem "_" begin
    import Aqua
    Aqua.test_all(ZarrZfp; ambiguities=false)
    Aqua.test_ambiguities(ZarrZfp)

    import CompatHelperLocal as CHL
    CHL.@check()
end
