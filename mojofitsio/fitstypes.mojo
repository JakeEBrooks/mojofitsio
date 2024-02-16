from collections.dict import Dict, KeyElement

struct PrimaryHDU[T: DType]: # Can't be a collection element yet until Dict implements __copyinit__
    var header: Header
    var data: Tensor[T]

    fn __init__(inout self, owned header: Header, data: Tensor[T]):
        self.header = header^
        self.data = data
    
    fn __moveinit__(inout self, owned existing: PrimaryHDU[T]):
        self.header = existing.header^
        self.data = existing.data^

struct Header:
    var hdict: Dict[HeaderKey, String]

    fn __init__(inout self):
        self.hdict = Dict[HeaderKey, String]()

    fn __init__(inout self, owned in_dict: Dict[HeaderKey, String]):
        self.hdict = in_dict^
    
    fn __getitem__(self, key: HeaderKey) raises -> String:
        return self.hdict[key]
    
    fn __setitem__(inout self, key: HeaderKey, value: String):
        self.hdict[key] = value
    
    fn __moveinit__(inout self, owned existing: Header):
        self.hdict = existing.hdict^
    
    fn __contains__(self, key: HeaderKey) -> Bool:
        return self.hdict.__contains__(key)
    
    fn __len__(self) -> Int:
        return self.hdict.__len__()
    
    fn __del__(owned self):
        self.hdict^.__del__()

@value
struct HeaderKey(KeyElement):
    var s: String

    fn __init__(inout self, owned s: String):
        self.s = s^

    fn __init__(inout self, s: StringLiteral):
        self.s = String(s)

    fn __hash__(self) -> Int:
        return hash(self.s)

    fn __eq__(self, other: Self) -> Bool:
        return self.s == other.s
