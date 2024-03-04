from tensor import TensorShape
from sys.info import is_big_endian
from math.bit import bswap
from memory.unsafe import bitcast
from algorithm import vectorize

from .config import config

fn get_headers_start_blocks(file: FileHandle) raises -> DynamicVector[Int]:
    """
    Construct a DynamicVector of integers representing the index of the first FITS block in each header.
    """
    var bounds = DynamicVector[Int]()
    var bytecounter = SIMD[DType.uint64, 1](0)
    var teststring: String = "SIMPLE "
    while teststring != "":
        bytecounter = file.seek(bytecounter)
        teststring = file.read(config.header_keyword_length)
        if "SIMPLE" in teststring or "XTENSION" in teststring:
            bounds.append(int(bytecounter)//config.fits_block_length)
        bytecounter += config.fits_block_length
    return bounds

fn get_header_sizes(file: FileHandle, startblocks: DynamicVector[Int]) raises -> DynamicVector[Int]:
    """
    Construct a DynamicVector of integers representing the FITS block size of each header.
    """
    var sizes = DynamicVector[Int]()
    var blockcounter: Int
    var foundEND: Bool = False
    for i in range(len(startblocks)):
        blockcounter = 1
        foundEND = False
        while not foundEND:
            if _block_has_END(get_FITS_blocks(file, startblocks[i]+blockcounter)):
                foundEND = True
                sizes.append(blockcounter)
            else:
                blockcounter += 1
    return sizes

fn get_data_sizes(file: FileHandle, headerstartblocks: DynamicVector[Int], headersizes: DynamicVector[Int]) raises -> DynamicVector[Int]:
    """
    Construct a DynamicVector of integers representing the FITS block size of each data unit.
    """
    var data_sizes = DynamicVector[Int]()
    for i in range(len(headerstartblocks)-1):
        data_sizes.append(headerstartblocks[i+1] - (headerstartblocks[i] + headersizes[i]))
    
    var testblock = Tensor[DType.int8](config.fits_block_length)
    var blockcounter: Int = 0
    var foundEOF: Bool = False
    while not foundEOF:
        if get_FITS_blocks(file, headerstartblocks[-1]+headersizes[-1]+blockcounter).num_elements() != 0:
            blockcounter += 1
        else:
            foundEOF = True
    data_sizes.append(blockcounter)
    return data_sizes

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

fn get_header_line(buff: Tensor[DType.int8], line: Int) -> String:
    """
    Return a full header entry as a String.
    """
    var ptr = DTypePointer[DType.int8].alloc(config.header_line_length)
    @parameter
    fn store_bytes[simd_width: Int](index: Int):
        ptr.simd_store[simd_width](index, buff.data().simd_load[simd_width](line*config.header_line_length+index))
    vectorize[store_bytes, simdwidthof[DType.int8]()](config.header_line_length)
    return StringRef(ptr, config.header_line_length)

fn get_header_keyword(buff: Tensor[DType.int8], line: Int) -> String: # Hard coded to 8 byte keyword to remove any overhead from `vectorize`
    """
    Return the keyword portion of a header entry as a String.
    """
    var ptr = DTypePointer[DType.int8].alloc(8)
    ptr.simd_store[8](0, buff.data().simd_load[8](line*config.header_line_length))
    return StringRef(ptr, 8)

fn get_header_valueind(buff: Tensor[DType.int8], line: Int) -> String: # hard coded to 2 byte value indicator to remove any overhead from `vectorize`
    """
    Return the value indicator portion of a header entry as a String.
    """
    var ptr = DTypePointer[DType.int8].alloc(2)
    ptr.simd_store[2](0, buff.data().simd_load[2](line*config.header_line_length+8))
    return StringRef(ptr, 2)

fn get_header_field(buff: Tensor[DType.int8], line: Int) -> String:
    """
    Return the field portion of a header entry as a String.
    """
    var ptr = DTypePointer[DType.int8].alloc(config.header_field_length)
    @parameter
    fn store_bytes[simd_width: Int](index: Int):
        ptr.simd_store[simd_width](index, buff.data().simd_load[simd_width](line*config.header_line_length+config.header_keyword_length+config.header_valueind_length+index))
    vectorize[store_bytes, simdwidthof[DType.int8]()](config.header_field_length)
    return StringRef(ptr, config.header_field_length)

fn _construct_i16_image(buff: Tensor[DType.int8], shape: TensorShape) -> Tensor[DType.int16]:
    """
    Construct a Tensor of 16-bit integers from an integral number of FITS blocks representing a FITS Image.
    """
    alias i8simdwidth = simdwidthof[DType.int8]()
    alias i16simdwidth = simdwidthof[DType.int16]()

    var out = Tensor[DType.int16](shape)
    var iters = shape.num_elements()//i16simdwidth
    for i in range(0,iters):
        out.data().simd_store[i16simdwidth](i*i16simdwidth, _i16_frombytes[i8simdwidth, i16simdwidth](buff.data().simd_load[i8simdwidth](i*i8simdwidth)))
    if iters*i16simdwidth < shape.num_elements():
        for i in range(iters*i16simdwidth, shape.num_elements()):
            out.data().store(i, _i16_frombytes[2, 1](buff.data().simd_load[2](i*2)))
    return out

fn _construct_i32_image(buff: Tensor[DType.int8], shape: TensorShape) -> Tensor[DType.int32]:
    """
    Construct a Tensor of 32-bit integers from an integral number of FITS blocks representing a FITS Image.
    """
    alias i8simdwidth = simdwidthof[DType.int8]()
    alias i32simdwidth = simdwidthof[DType.int32]()

    var out = Tensor[DType.int32](shape)
    var iters = shape.num_elements()//i32simdwidth
    for i in range(0,iters):
        out.data().simd_store[i32simdwidth](i*i32simdwidth, _i32_frombytes[i8simdwidth, i32simdwidth](buff.data().simd_load[i8simdwidth](i*i8simdwidth)))
    if iters*i32simdwidth < shape.num_elements():
        for i in range(iters*i32simdwidth, shape.num_elements()):
            out.data().store(i, _i32_frombytes[4, 1](buff.data().simd_load[4](i*4)))
    return out

fn _construct_i64_image(buff: Tensor[DType.int8], shape: TensorShape) -> Tensor[DType.int64]:
    """
    Construct a Tensor of 64-bit integers from an integral number of FITS blocks representing a FITS Image.
    """
    alias i8simdwidth = simdwidthof[DType.int8]()
    alias i64simdwidth = simdwidthof[DType.int64]()

    var out = Tensor[DType.int64](shape)
    var iters = shape.num_elements()//i64simdwidth
    for i in range(0,iters):
        out.data().simd_store[i64simdwidth](i*i64simdwidth, _i64_frombytes[i8simdwidth, i64simdwidth](buff.data().simd_load[i8simdwidth](i*i8simdwidth)))
    if iters*i64simdwidth < shape.num_elements():
        for i in range(iters*i64simdwidth, shape.num_elements()):
            out.data().store(i, _i64_frombytes[8, 1](buff.data().simd_load[8](i*8)))
    return out

fn _construct_f32_image(buff: Tensor[DType.int8], shape: TensorShape) -> Tensor[DType.float32]:
    """
    Construct a Tensor of 32-bit floats from an integral number of FITS blocks representing a FITS Image.
    """
    alias i8simdwidth = simdwidthof[DType.int8]()
    alias f32simdwidth = simdwidthof[DType.float32]()

    var out = Tensor[DType.float32](shape)
    var iters = shape.num_elements()//f32simdwidth
    for i in range(0,iters):
        out.data().simd_store[f32simdwidth](i*f32simdwidth, _f32_frombytes[i8simdwidth, f32simdwidth](buff.data().simd_load[i8simdwidth](i*i8simdwidth)))
    if iters*f32simdwidth < shape.num_elements():
        for i in range(iters*f32simdwidth, shape.num_elements()):
            out.data().store(i, _f32_frombytes[4, 1](buff.data().simd_load[4](i*4)))
    return out

fn _construct_f64_image(buff: Tensor[DType.int8], shape: TensorShape) -> Tensor[DType.float64]:
    """
    Construct a Tensor of 64-bit floats from an integral number of FITS blocks representing a FITS Image.
    """
    alias i8simdwidth = simdwidthof[DType.int8]()
    alias f64simdwidth = simdwidthof[DType.float64]()

    var out = Tensor[DType.float64](shape)
    var iters = shape.num_elements()//f64simdwidth
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
    while not foundEND and bytecounter < config.fits_block_length:
        if buff.simd_load[4](bytecounter) == SIMD[DType.int8, 4](69, 78, 68, 32): # "END "
            foundEND = True
        else:
            bytecounter += config.header_line_length
    return foundEND

fn _i16_frombytes[inwidth: Int, outwidth: Int](data: SIMD[DType.int8, inwidth]) -> SIMD[DType.int16, outwidth]:
    """
    Converts a SIMD of bytes to a SIMD of 16-bit integers.
    """
    @parameter
    if is_big_endian():
        return bswap(bitcast[DType.int16, outwidth](data))
    else:
        return bitcast[DType.int16, outwidth](data)

fn _i32_frombytes[inwidth: Int, outwidth: Int](data: SIMD[DType.int8, inwidth]) -> SIMD[DType.int32, outwidth]:
    """
    Converts a SIMD of bytes to a SIMD of 32-bit integers.
    """
    @parameter
    if is_big_endian():
        return bswap(bitcast[DType.int32, outwidth](data))
    else:
        return bitcast[DType.int32, outwidth](data)

fn _i64_frombytes[inwidth: Int, outwidth: Int](data: SIMD[DType.int8, inwidth]) -> SIMD[DType.int64, outwidth]:
    """
    Converts a SIMD of bytes to a SIMD of 64-bit integers.
    """
    @parameter
    if is_big_endian():
        return bswap(bitcast[DType.int64, outwidth](data))
    else:
        return bitcast[DType.int64, outwidth](data)

fn _f32_frombytes[inwidth: Int, outwidth: Int](data: SIMD[DType.int8, inwidth]) -> SIMD[DType.float32, outwidth]:
    """
    Converts a SIMD of bytes to a SIMD of 32-bit floats.
    """
    var intdata = bitcast[DType.int32, outwidth](data)
    @parameter
    if is_big_endian():
        return bitcast[DType.float32, outwidth](bswap(intdata))
    else:
        return bitcast[DType.float32, outwidth](intdata)

fn _f64_frombytes[inwidth: Int, outwidth: Int](data: SIMD[DType.int8, inwidth]) -> SIMD[DType.float64, outwidth]:
    """
    Converts a SIMD of bytes to a SIMD of 64-bit floats.
    """
    var intdata = bitcast[DType.int64, outwidth](data)
    @parameter
    if is_big_endian():
        return bitcast[DType.float64, outwidth](bswap(intdata))
    else:
        return bitcast[DType.float64, outwidth](intdata)
