from tensor import TensorShape
from math.bit import bswap
from memory.unsafe import bitcast

from .config import iobounds
from .fitstypes import PrimaryHDU, Header
from .byteio import _i16_frombytes, _i32_frombytes, _i64_frombytes, _f32_frombytes, _f64_frombytes

# fn open() -> HDUL: ...

fn get_i16PrimaryData(file: FileHandle, shape: TensorShape) raises -> Tensor[DType.int16]:
    var foundEND: Bool = False
    var blockcount: Int = 0
    while not foundEND:
        if _block_has_END(_read_FITS_blocks(file, blockcount)):
            foundEND = True
            blockcount += 1
        else:
            blockcount += 1

    return _construct_i16_image(_get_data_blocks(file, blockcount, 16, shape), shape)

fn get_f32PrimaryData(file: FileHandle, shape: TensorShape) raises -> Tensor[DType.float32]:
    var foundEND: Bool = False
    var blockcount: Int = 0
    while not foundEND:
        if _block_has_END(_read_FITS_blocks(file, blockcount)):
            foundEND = True
            blockcount += 1
        else:
            blockcount += 1

    return _construct_f32_image(_get_data_blocks(file, blockcount, 32, shape), shape)

fn _read_FITS_blocks(file: FileHandle, startblock: Int) raises -> Tensor[DType.int8]:
    let bc = file.seek(iobounds.fits_block_length*startblock)
    return file.read_bytes(iobounds.fits_block_length)

fn _read_FITS_blocks(file: FileHandle, startblock: Int, endblock: Int) raises -> Tensor[DType.int8]:
    let bc = file.seek(iobounds.fits_block_length*startblock)
    return file.read_bytes(iobounds.fits_block_length*(endblock - startblock))

fn _construct_header(buff: Tensor[DType.int8]) -> Header: ... #return a header

fn _get_data_blocks(file: FileHandle, startblock: Int, bitpix: Int, shape: TensorShape) raises -> Tensor[DType.int8]:
    let total_blocks = math.ceildiv(bitpix//8 * shape.num_elements(), iobounds.fits_block_length)
    return _read_FITS_blocks(file, startblock, startblock+total_blocks)

fn _construct_i16_image(buff: Tensor[DType.int8], shape: TensorShape) -> Tensor[DType.int16]:
    alias i8simdwidth = simdwidthof[DType.int8]()
    alias i16simdwidth = simdwidthof[DType.int16]()

    var out = Tensor[DType.int16](shape)
    let iters = shape.num_elements()//i16simdwidth
    for i in range(0,iters):
        out.data().simd_store[i16simdwidth](i*i16simdwidth, _i16_frombytes[i8simdwidth, i16simdwidth](buff.data().simd_load[i8simdwidth](i*i8simdwidth)))
    if iters*i16simdwidth < shape.num_elements():
        for i in range(iters*i16simdwidth, shape.num_elements()):
            out.data().store(i, _i16_frombytes[2, 1](buff.data().simd_load[2](i*2)))
    return out

fn _construct_i32_image(buff: Tensor[DType.int8], shape: TensorShape) -> Tensor[DType.int32]:
    alias i8simdwidth = simdwidthof[DType.int8]()
    alias i32simdwidth = simdwidthof[DType.int32]()

    var out = Tensor[DType.int32](shape)
    let iters = shape.num_elements()//i32simdwidth
    for i in range(0,iters):
        out.data().simd_store[i32simdwidth](i*i32simdwidth, _i32_frombytes[i8simdwidth, i32simdwidth](buff.data().simd_load[i8simdwidth](i*i8simdwidth)))
    if iters*i32simdwidth < shape.num_elements():
        for i in range(iters*i32simdwidth, shape.num_elements()):
            out.data().store(i, _i32_frombytes[4, 1](buff.data().simd_load[4](i*4)))
    return out

fn _construct_i64_image(buff: Tensor[DType.int8], shape: TensorShape) -> Tensor[DType.int64]:
    alias i8simdwidth = simdwidthof[DType.int8]()
    alias i64simdwidth = simdwidthof[DType.int64]()

    var out = Tensor[DType.int64](shape)
    let iters = shape.num_elements()//i64simdwidth
    for i in range(0,iters):
        out.data().simd_store[i64simdwidth](i*i64simdwidth, _i64_frombytes[i8simdwidth, i64simdwidth](buff.data().simd_load[i8simdwidth](i*i8simdwidth)))
    if iters*i64simdwidth < shape.num_elements():
        for i in range(iters*i64simdwidth, shape.num_elements()):
            out.data().store(i, _i64_frombytes[8, 1](buff.data().simd_load[8](i*8)))
    return out

fn _construct_f32_image(buff: Tensor[DType.int8], shape: TensorShape) -> Tensor[DType.float32]:
    alias i8simdwidth = simdwidthof[DType.int8]()
    alias f32simdwidth = simdwidthof[DType.float32]()

    var out = Tensor[DType.float32](shape)
    let iters = shape.num_elements()//f32simdwidth
    for i in range(0,iters):
        out.data().simd_store[f32simdwidth](i*f32simdwidth, _f32_frombytes[i8simdwidth, f32simdwidth](buff.data().simd_load[i8simdwidth](i*i8simdwidth)))
    if iters*f32simdwidth < shape.num_elements():
        for i in range(iters*f32simdwidth, shape.num_elements()):
            out.data().store(i, _f32_frombytes[4, 1](buff.data().simd_load[4](i*4)))
    return out

fn _construct_f64_image(buff: Tensor[DType.int8], shape: TensorShape) -> Tensor[DType.float64]:
    alias i8simdwidth = simdwidthof[DType.int8]()
    alias f64simdwidth = simdwidthof[DType.float64]()

    var out = Tensor[DType.float64](shape)
    let iters = shape.num_elements()//f64simdwidth
    for i in range(0,iters):
        out.data().simd_store[f64simdwidth](i*f64simdwidth, _f64_frombytes[i8simdwidth, f64simdwidth](buff.data().simd_load[i8simdwidth](i*i8simdwidth)))
    if iters*f64simdwidth < shape.num_elements():
        for i in range(iters*f64simdwidth, shape.num_elements()):
            out.data().store(i, _f64_frombytes[8, 1](buff.data().simd_load[8](i*8)))
    return out

fn _block_has_END(buff: Tensor[DType.int8]) -> Bool:
    """
    Check to see if a FITS block has the END header keyword in it.
    """
    var foundEND: Bool = False
    var bytecounter: Int = 0
    let END = SIMD[DType.int8, 4](69, 78, 68, 32)
    while not foundEND and bytecounter < iobounds.fits_block_length:
        if buff.simd_load[4](bytecounter) == END:
            foundEND = True
        else:
            bytecounter += iobounds.header_line_length
    return foundEND
