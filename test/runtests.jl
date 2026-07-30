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


@testitem "_" begin
    import Aqua
    Aqua.test_all(ZarrZfp; ambiguities=false)
    Aqua.test_ambiguities(ZarrZfp)

    import CompatHelperLocal as CHL
    CHL.@check()
end
