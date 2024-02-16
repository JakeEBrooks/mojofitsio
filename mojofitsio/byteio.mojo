from memory.unsafe import bitcast
from math.bit import bswap

fn _i16_frombytes[inwidth: Int, outwidth: Int](data: SIMD[DType.int8, inwidth], is_big_endian: Bool = True) -> SIMD[DType.int16, outwidth]:
    if is_big_endian:
        return bswap(bitcast[DType.int16, outwidth](data))
    else:
        return bitcast[DType.int16, outwidth](data)

fn _i32_frombytes[inwidth: Int, outwidth: Int](data: SIMD[DType.int8, inwidth], is_big_endian: Bool = True) -> SIMD[DType.int32, outwidth]:
    if is_big_endian:
        return bswap(bitcast[DType.int32, outwidth](data))
    else:
        return bitcast[DType.int32, outwidth](data)

fn _i64_frombytes[inwidth: Int, outwidth: Int](data: SIMD[DType.int8, inwidth], is_big_endian: Bool = True) -> SIMD[DType.int64, outwidth]:
    if is_big_endian:
        return bswap(bitcast[DType.int64, outwidth](data))
    else:
        return bitcast[DType.int64, outwidth](data)

fn _f32_frombytes[inwidth: Int, outwidth: Int](data: SIMD[DType.int8, inwidth], is_big_endian: Bool = True) -> SIMD[DType.float32, outwidth]:
    let intdata = bitcast[DType.int32, outwidth](data)
    if is_big_endian:
        return bitcast[DType.float32, outwidth](bswap(intdata))
    else:
        return bitcast[DType.float32, outwidth](intdata)

fn _f64_frombytes[inwidth: Int, outwidth: Int](data: SIMD[DType.int8, inwidth], is_big_endian: Bool = True) -> SIMD[DType.float64, outwidth]:
    let intdata = bitcast[DType.int64, outwidth](data)
    if is_big_endian:
        return bitcast[DType.float64, outwidth](bswap(intdata))
    else:
        return bitcast[DType.float64, outwidth](intdata)
