from collections.dict import Dict

struct Header: # This isn't ta final implementation, until Dict gets improved it'll do
    var hdict: Dict[HeaderKey, String]
    var keys: DynamicVector[HeaderKey]

    fn __init__(inout self):
        self.hdict = Dict[HeaderKey, String]()
        self.keys = DynamicVector[HeaderKey]()
    
    fn __getitem__(self, key: HeaderKey) raises -> String:
        return self.hdict[key]
    
    fn __setitem__(inout self, key: HeaderKey, value: String):
        if not self.hdict.__contains__(key):
            self.keys.append(key)
        self.hdict[key] = value
    
    fn __copyinit__(inout self, existing: Self):
        self.hdict = Dict[HeaderKey, String]()
        self.keys = existing.keys
        try:
            for i in range(len(existing.keys)):
                self.hdict[existing.keys[i]] = existing.hdict[existing.keys[i]]
        except:
            pass
    
    fn __moveinit__(inout self, owned existing: Self):
        self.hdict = existing.hdict^
        self.keys = existing.keys^
    
    fn __contains__(self, key: HeaderKey) -> Bool:
        return self.hdict.__contains__(key)
    
    fn __len__(self) -> Int:
        return self.hdict.__len__()
    
    fn __del__(owned self):
        self.hdict^.__del__()

@value
struct HeaderKey(KeyElement, Stringable):
    var s: String

    fn __init__(inout self, owned s: String):
        self.s = s^

    fn __init__(inout self, s: StringLiteral):
        self.s = String(s)

    fn __hash__(self) -> Int:
        return hash(self.s)
    
    fn __str__(self) -> String:
        return self.s

    fn __eq__(self, other: Self) -> Bool:
        return self.s == other.s