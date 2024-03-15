from sys.info import is_big_endian
from math.bit import bswap

fn convert_byteSIMD[outtype: DType, inwidth: Int, outwidth: Int = inwidth//outtype.sizeof()](bytes: SIMD[DType.int8, inwidth]) -> SIMD[outtype, outwidth]:
    """
    Convert a `SIMD` of bytes to the desired `DType`. Assumes the endianness of the system.
    """
    var out: SIMD[outtype, outwidth]
    @parameter
    if is_big_endian():
        out = bswap(bitcast[outtype, outwidth](bytes))
    else:
        out = bitcast[outtype, outwidth](bytes)
    return out

fn convert_byteSIMD[outtype: DType, inwidth: Int, outwidth: Int = inwidth//outtype.sizeof()](bytes: SIMD[DType.int8, inwidth], big_endian: Bool) -> SIMD[outtype, outwidth]:
    """
    Convert a `SIMD` of bytes to the desired `DType` using the supplied endianness.
    """
    var out: SIMD[outtype, outwidth]
    if big_endian:
        out = bswap(bitcast[outtype, outwidth](bytes))
    else:
        out = bitcast[outtype, outwidth](bytes)
    return out