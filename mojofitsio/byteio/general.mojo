from sys.info import is_big_endian
from math.bit import bswap

from mojofitsio.config import config

fn get_FITS_blocks(file: FileHandle, startblock: Int) raises -> Tensor[DType.int8]:
    """
    Return the FITS block at an offset of *startblock* FITS blocks.
    """
    var bc = file.seek(config.fits_block_length*startblock)
    return file.read_bytes(config.fits_block_length)

fn get_FITS_blocks(file: FileHandle, startblock: Int, endblock: Int) raises -> Tensor[DType.int8]:
    """
    Return an integral number of FITS blocks in the range [*startblock*, *endblock*).
    """
    var bc = file.seek(config.fits_block_length*startblock)
    return file.read_bytes(config.fits_block_length*(endblock - startblock))

fn i16_frombytes[inwidth: Int, outwidth: Int = inwidth//2](data: SIMD[DType.int8, inwidth]) -> SIMD[DType.int16, outwidth]:
    """
    Converts a SIMD of bytes to a SIMD of 16-bit integers.
    """
    @parameter
    if is_big_endian():
        return bswap(bitcast[DType.int16, outwidth](data))
    else:
        return bitcast[DType.int16, outwidth](data)

fn i32_frombytes[inwidth: Int, outwidth: Int = inwidth//4](data: SIMD[DType.int8, inwidth]) -> SIMD[DType.int32, outwidth]:
    """
    Converts a SIMD of bytes to a SIMD of 32-bit integers.
    """
    @parameter
    if is_big_endian():
        return bswap(bitcast[DType.int32, outwidth](data))
    else:
        return bitcast[DType.int32, outwidth](data)

fn i64_frombytes[inwidth: Int, outwidth: Int = inwidth//8](data: SIMD[DType.int8, inwidth]) -> SIMD[DType.int64, outwidth]:
    """
    Converts a SIMD of bytes to a SIMD of 64-bit integers.
    """
    @parameter
    if is_big_endian():
        return bswap(bitcast[DType.int64, outwidth](data))
    else:
        return bitcast[DType.int64, outwidth](data)

fn f32_frombytes[inwidth: Int, outwidth: Int = inwidth//4](data: SIMD[DType.int8, inwidth]) -> SIMD[DType.float32, outwidth]:
    """
    Converts a SIMD of bytes to a SIMD of 32-bit floats.
    """
    var intdata = bitcast[DType.int32, outwidth](data)
    @parameter
    if is_big_endian():
        return bitcast[DType.float32, outwidth](bswap(intdata))
    else:
        return bitcast[DType.float32, outwidth](intdata)

fn f64_frombytes[inwidth: Int, outwidth: Int = inwidth//8](data: SIMD[DType.int8, inwidth]) -> SIMD[DType.float64, outwidth]:
    """
    Converts a SIMD of bytes to a SIMD of 64-bit floats.
    """
    var intdata = bitcast[DType.int64, outwidth](data)
    @parameter
    if is_big_endian():
        return bitcast[DType.float64, outwidth](bswap(intdata))
    else:
        return bitcast[DType.float64, outwidth](intdata)
