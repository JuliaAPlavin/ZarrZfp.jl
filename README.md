# ZarrZfp.jl

Exactly what it says on the tin: `zfp`-based compressor for `Zarr.jl`.

```julia
zcreate(Float32, ..., compressor=ZfpCompressor(rate=16))
```
