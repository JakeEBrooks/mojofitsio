from tensor import TensorShape

from mojofitsio.byteio.general import (
    i16_frombytes,
    i32_frombytes,
    i64_frombytes,
    f32_frombytes,
    f64_frombytes)


fn construct_i16_image(buff: Tensor[DType.int8], shape: TensorShape) -> Tensor[DType.int16]:
    """
    Construct a Tensor of 16-bit integers from an integral number of FITS blocks representing a FITS Image.
    """
    alias i8simdwidth = simdwidthof[DType.int8]()
    alias i16simdwidth = simdwidthof[DType.int16]()

    var out = Tensor[DType.int16](shape)
    var iters = shape.num_elements()//i16simdwidth
    for i in range(0,iters):
        out.data().simd_store[i16simdwidth](i*i16simdwidth, i16_frombytes[i8simdwidth, i16simdwidth](buff.data().simd_load[i8simdwidth](i*i8simdwidth)))
    if iters*i16simdwidth < shape.num_elements():
        for i in range(iters*i16simdwidth, shape.num_elements()):
            out.data().store(i, i16_frombytes[2, 1](buff.data().simd_load[2](i*2)))
    return out

fn construct_i32_image(buff: Tensor[DType.int8], shape: TensorShape) -> Tensor[DType.int32]:
    """
    Construct a Tensor of 32-bit integers from an integral number of FITS blocks representing a FITS Image.
    """
    alias i8simdwidth = simdwidthof[DType.int8]()
    alias i32simdwidth = simdwidthof[DType.int32]()

    var out = Tensor[DType.int32](shape)
    var iters = shape.num_elements()//i32simdwidth
    for i in range(0,iters):
        out.data().simd_store[i32simdwidth](i*i32simdwidth, i32_frombytes[i8simdwidth, i32simdwidth](buff.data().simd_load[i8simdwidth](i*i8simdwidth)))
    if iters*i32simdwidth < shape.num_elements():
        for i in range(iters*i32simdwidth, shape.num_elements()):
            out.data().store(i, i32_frombytes[4, 1](buff.data().simd_load[4](i*4)))
    return out

fn construct_i64_image(buff: Tensor[DType.int8], shape: TensorShape) -> Tensor[DType.int64]:
    """
    Construct a Tensor of 64-bit integers from an integral number of FITS blocks representing a FITS Image.
    """
    alias i8simdwidth = simdwidthof[DType.int8]()
    alias i64simdwidth = simdwidthof[DType.int64]()

    var out = Tensor[DType.int64](shape)
    var iters = shape.num_elements()//i64simdwidth
    for i in range(0,iters):
        out.data().simd_store[i64simdwidth](i*i64simdwidth, i64_frombytes[i8simdwidth, i64simdwidth](buff.data().simd_load[i8simdwidth](i*i8simdwidth)))
    if iters*i64simdwidth < shape.num_elements():
        for i in range(iters*i64simdwidth, shape.num_elements()):
            out.data().store(i, i64_frombytes[8, 1](buff.data().simd_load[8](i*8)))
    return out

fn construct_f32_image(buff: Tensor[DType.int8], shape: TensorShape) -> Tensor[DType.float32]:
    """
    Construct a Tensor of 32-bit floats from an integral number of FITS blocks representing a FITS Image.
    """
    alias i8simdwidth = simdwidthof[DType.int8]()
    alias f32simdwidth = simdwidthof[DType.float32]()

    var out = Tensor[DType.float32](shape)
    var iters = shape.num_elements()//f32simdwidth
    for i in range(0,iters):
        out.data().simd_store[f32simdwidth](i*f32simdwidth, f32_frombytes[i8simdwidth, f32simdwidth](buff.data().simd_load[i8simdwidth](i*i8simdwidth)))
    if iters*f32simdwidth < shape.num_elements():
        for i in range(iters*f32simdwidth, shape.num_elements()):
            out.data().store(i, f32_frombytes[4, 1](buff.data().simd_load[4](i*4)))
    return out

fn construct_f64_image(buff: Tensor[DType.int8], shape: TensorShape) -> Tensor[DType.float64]:
    """
    Construct a Tensor of 64-bit floats from an integral number of FITS blocks representing a FITS Image.
    """
    alias i8simdwidth = simdwidthof[DType.int8]()
    alias f64simdwidth = simdwidthof[DType.float64]()

    var out = Tensor[DType.float64](shape)
    var iters = shape.num_elements()//f64simdwidth
    for i in range(0,iters):
        out.data().simd_store[f64simdwidth](i*f64simdwidth, f64_frombytes[i8simdwidth, f64simdwidth](buff.data().simd_load[i8simdwidth](i*i8simdwidth)))
    if iters*f64simdwidth < shape.num_elements():
        for i in range(iters*f64simdwidth, shape.num_elements()):
            out.data().store(i, f64_frombytes[8, 1](buff.data().simd_load[8](i*8)))
    return out
