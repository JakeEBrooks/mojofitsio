from mojofitsio.header import Header

trait IsHDUType(CollectionElement):
    fn header(self) -> Header: ...
    fn datatype(self) -> FITSDataType: ...
    fn HDUtype(self) -> HDUType: ...

struct HDU[T: IsHDUType]:
    var header: Header
    var data: T
    var hdutype: HDUType
    var dtype: FITSDataType

    fn __init__(inout self, header: Header, data: T):
        self.header = header
        self.data = data
        self.hdutype = data.HDUtype()
        self.dtype = data.datatype()

@register_passable("trivial")
struct HDUType(Stringable):
    var value: UInt8

    alias EMPTY = HDUType(0)
    alias IMAGE = HDUType(1)
    alias TABLE = HDUType(2)
    alias BINTABLE = HDUType(3)
    alias ANY = HDUType(4)

    fn __init__(inout self, value: UInt8):
        self.value = value

    fn __eq__(self, existing: Self) -> Bool:
        return self.value == existing.value

    fn __str__(self) -> String:
        var outstr: String = ""
        if self.value == 0:
            outstr = "EMPTY"
        elif self.value == 1:
            outstr = "IMAGE"
        elif self.value == 2:
            outstr = "TABLE"
        elif self.value == 3:
            outstr = "BINTABLE"
        elif self.value == 4:
            outstr = "ANY"
        return outstr
    
    @staticmethod
    fn from_string(field: String) -> Self:
        var outtype: Self = Self.EMPTY
        if "'IMAGE   '" in field:
            outtype = Self.IMAGE
        elif "'TABLE   '" in field:
            outtype = Self.TABLE
        elif "'BINTABLE'" in field:
            outtype = Self.BINTABLE
        return outtype

@register_passable("trivial")
struct FITSDataType(Stringable):
    var value: Int8

    alias UI8 = FITSDataType(8)
    alias SI16 = FITSDataType(16)
    alias SI32 = FITSDataType(32)
    alias SI64 = FITSDataType(64)
    alias F32 = FITSDataType(-32)
    alias F64 = FITSDataType(-64)

    fn __init__(inout self, value: Int8):
        self.value = value
    
    fn __eq__(self, existing: Self) -> Bool:
        return self.value == existing.value

    fn __str__(self) -> String:
        var out: String = ""
        if self.value == 8:
            out = "Unsigned 8-bit Integer"
        elif self.value == 16:
            out = "Signed 16-bit Integer"
        elif self.value == 32:
            out = "Signed 32-bit Integer"
        elif self.value == 64:
            out = "Signed 64-bit Integer"
        elif self.value == -32:
            out = "32-bit Float"
        elif self.value == -64:
            out = "64-bit Float"
        return out
    
    @always_inline
    fn bitpix(self) -> Int8:
        return self.value

    @staticmethod
    fn from_bitpix(bitpix: Int) raises -> Self:
        return Self(bitpix)

    @staticmethod
    fn from_bitpix(bitpix: String) raises -> Self:
        var to_int: Int = atol(bitpix)
        return Self.from_bitpix(to_int)

