from tensor import TensorShape


struct FITSData[T: FITSDType]:
    var data: Tensor[T.dtype]
    var fitstype: FITSType

    fn __init__(inout self, data: Tensor[T.dtype], fitstype: FITSType):
        self.data = data
        self.fitstype = fitstype
    
    fn __copyinit__(inout self, existing: Self):
        self.data = existing.data
        self.fitstype = existing.fitstype
    
    fn __moveinit__(inout self, owned existing: Self):
        self.data = existing.data^
        self.fitstype = existing.fitstype


@register_passable("trivial")
struct FITSDType(Stringable):
    """
    A wrapper around DType that restricts possible values to those acceptable in the
    FITS format. Also designed to interact with DType as normal and go to and from values of BITPIX.
    """
    var dtype: DType

    alias uint8 = FITSDType{dtype: DType.uint8}
    alias int16 = FITSDType{dtype: DType.int16}
    alias int32 = FITSDType{dtype: DType.int32}
    alias int64 = FITSDType{dtype: DType.int64}
    alias float32 = FITSDType{dtype: DType.float32}
    alias float64 = FITSDType{dtype: DType.float64}

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        return self.dtype == other.dtype
    @always_inline
    fn __eq__(self, other: DType) -> Bool:
        return self.dtype == other

    @always_inline
    fn __neq__(self, other: Self) -> Bool:
        return self.dtype != other.dtype
    @always_inline
    fn __neq__(self, other: DType) -> Bool:
        return self.dtype != other
    
    @always_inline
    fn __str__(self) -> String:
        return self.dtype.__str__()
    
    @always_inline
    fn isa[other: Self](self) -> Bool:
        return self.dtype.isa[other.dtype]()
    @always_inline
    fn isa[other: DType](self) -> Bool:
        return self.dtype.isa[other]()
    
    @always_inline
    fn sizeof(self) -> Int:
        return self.dtype.sizeof()
    
    @always_inline
    fn bitwidth(self) -> Int:
        return self.dtype.bitwidth()
    
    @staticmethod
    fn is_valid_bitpix(bitpix: Int) -> Bool:
        if (bitpix == 8 or bitpix == 16 or bitpix == 32 
            or bitpix == 64 or bitpix == -32 or bitpix == -64):
            return True
        else:
            return False
    
    @staticmethod
    fn from_bitpix(bitpix: Int) raises -> Self:
        var outtype: Self
        if bitpix == 8:
            outtype = FITSDType.uint8
        elif bitpix == 16:
            outtype = FITSDType.int16
        elif bitpix == 32:
            outtype = FITSDType.int32
        elif bitpix == 64:
            outtype = FITSDType.int64
        elif bitpix == -32:
            outtype = FITSDType.float32
        elif bitpix == -64:
            outtype = FITSDType.float64
        else:
            raise Error("Invalid value of BITPIX supplied to FITSDType")
        return outtype
    
    @staticmethod
    fn from_bitpix(bitpix: String) raises -> Self:
        return Self.from_bitpix(atol(bitpix))

@register_passable("trivial")
struct FITSType(Stringable):
    """
    A type representing the FITS type of a data structure, e.g. whether it is an IMAGE or a TABLE. Not to be confused with `FITSDType`.
    """
    var value: UInt8

    alias EMPTY = FITSType{value: 0}
    alias IMAGE = FITSType{value: 1}
    alias TABLE = FITSType{value: 2}
    alias BINTABLE = FITSType{value: 3}
    
    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    @always_inline    
    fn __neq__(self, other: Self) -> Bool:
        return self.value != other.value
    
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
        return outstr

    @staticmethod
    fn from_headerfield(field: String) -> Self:
        var outtype: Self = Self.EMPTY
        if "'IMAGE   '" in field:
            outtype = Self.IMAGE
        elif "'TABLE   '" in field:
            outtype = Self.TABLE
        elif "'BINTABLE'" in field:
            outtype = Self.BINTABLE
        return outtype
    
