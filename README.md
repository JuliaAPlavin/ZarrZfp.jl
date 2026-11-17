# ZarrZfp.jl

Exactly what it says on the tin: `zfp`-based compressor for `Zarr.jl`.

Zarr v2 (compressor `zfpy`):
```julia
zcreate(Float32, dims...; compressor = ZfpCompressor(rate = 16))
```

Zarr v3 (codec `numcodecs.zfpy`, like in zarr-python):
```julia
zcreate(Float32, dims...; zarr_format = 3, codecs = (ZfpyCodec(rate = 16),))
```

Both accept one of `tol`, `precision`, `rate`, or nothing for lossless, and produce identical chunk bytes.
