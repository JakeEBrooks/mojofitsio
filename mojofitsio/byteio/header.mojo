from algorithm import vectorize

from mojofitsio.config import config

fn get_header_line(buff: Tensor[DType.int8], line: UInt64) -> String:
    """
    Given a `Tensor` of bytes representing a header, return a full header entry as a String.
    """
    var ptr = DTypePointer[DType.int8].alloc(config.header_line_length)
    @parameter
    fn store_bytes[simd_width: Int](index: Int):
        ptr.simd_store[simd_width](index, buff.data().simd_load[simd_width](line*config.header_line_length+index))
    vectorize[store_bytes, simdwidthof[DType.int8]()](config.header_line_length)
    return StringRef(ptr, config.header_line_length)

fn get_header_keyword(buff: Tensor[DType.int8], line: UInt64) -> String: # Hard coded to 8 byte keyword to remove any overhead from `vectorize`
    """
    Given a `Tensor` of bytes representing a header, return the keyword portion of a header entry as a String.
    """
    var ptr = DTypePointer[DType.int8].alloc(8)
    ptr.simd_store[8](0, buff.data().simd_load[8](line*config.header_line_length))
    return StringRef(ptr, 8)

fn get_header_valueind(buff: Tensor[DType.int8], line: UInt64) -> String: # hard coded to 2 byte value indicator to remove any overhead from `vectorize`
    """
    Given a `Tensor` of bytes representing a header, return the value indicator portion of a header entry as a String.
    """
    var ptr = DTypePointer[DType.int8].alloc(2)
    ptr.simd_store[2](0, buff.data().simd_load[2](line*config.header_line_length+8))
    return StringRef(ptr, 2)

fn get_header_field(buff: Tensor[DType.int8], line: UInt64) -> String:
    """
    Given a `Tensor` of bytes representing a header, return the field portion of a header entry as a String.
    """
    var ptr = DTypePointer[DType.int8].alloc(config.header_field_length)
    @parameter
    fn store_bytes[simd_width: Int](index: Int):
        ptr.simd_store[simd_width](index, buff.data().simd_load[simd_width](line*config.header_line_length+config.header_keyword_length+config.header_valueind_length+index))
    vectorize[store_bytes, simdwidthof[DType.int8]()](config.header_field_length)
    return StringRef(ptr, config.header_field_length)

fn get_header_field_nocomment(buff: Tensor[DType.int8], line: UInt64) -> String:
    """
    Given a `Tensor` of bytes representing a header, return the field portion of a header entry as a String. The returned string
    will not contain the comment portion of the field if any was present.
    """
    # Locate the '/' character
    var nocomment_fieldlength: Int = config.header_field_length
    for i in range(config.header_field_length):
        if buff[int(line)*config.header_line_length+config.header_keyword_length+config.header_valueind_length+i] == 47:
            nocomment_fieldlength = i
    # Now create the string
    var ptr = DTypePointer[DType.int8].alloc(nocomment_fieldlength)
    @parameter
    fn store_bytes[simd_width: Int](index: Int):
        ptr.simd_store[simd_width](index, buff.data().simd_load[simd_width](line*config.header_line_length+config.header_keyword_length+config.header_valueind_length+index))
    vectorize[store_bytes, simdwidthof[DType.int8]()](nocomment_fieldlength)
    return StringRef(ptr, nocomment_fieldlength)



