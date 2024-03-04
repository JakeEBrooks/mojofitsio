struct FITSType(Stringable):
    var value: UInt8

    alias EMPTY = FITSType(0)
    alias IMAGE = FITSType(1)
    alias TABLE = FITSType(2)
    alias BINTABLE = FITSType(3)

    fn __init__(inout self, value: UInt8):
        self.value = value

    fn __copyinit__(inout self, existing: Self):
        self.value = existing.value

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
        return outstr

struct DataUnit[T: DType]:
    var data: Tensor[T]
    var type: FITSType

    fn __init__(inout self):
        self.data = Tensor[T]()
        self.type = FITSType.EMPTY
    
    fn __init__(inout self, data: Tensor[T], type: FITSType):
        self.data = data
        self.type = type

