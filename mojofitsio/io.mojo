from tensor import TensorShape

from mojofitsio.config import config
from mojofitsio.header import Header
from mojofitsio.fitstypes import FITSDataType, HDUType

struct FITSMap:
    var hdu_maps: DynamicVector[HDUMap]
    var num_hdu: UInt64

    fn __init__(inout self):
        self.hdu_maps = DynamicVector[HDUMap]()
        self.num_hdu = 0

    fn __init__(inout self, *maps: HDUMap):
        self.hdu_maps = DynamicVector[HDUMap]()
        for i in range(len(maps)):
            self.hdu_maps.append(maps[i])
        self.num_hdu = len(self.hdu_maps)
    
    fn __getitem__(self, index: Int) -> HDUMap:
        return self.hdu_maps[index]
    
    fn __setitem__(inout self, index: Int, val: HDUMap):
        self.hdu_maps[index] = val
    
    fn __copyinit__(inout self, existing: Self):
        self.hdu_maps = existing.hdu_maps
        self.num_hdu = existing.num_hdu
    
    fn __moveinit__(inout self, owned existing: Self):
        self.hdu_maps = existing.hdu_maps^
        self.num_hdu = existing.num_hdu
    
    fn append(inout self, to_append: HDUMap):
        self.hdu_maps.append(to_append)
        self.num_hdu += 1
    
    fn blocksize(self) -> UInt64:
        var total: UInt64 = 0
        for i in range(self.num_hdu):
            total += self.hdu_maps[i].blocksize()
        return total
    
    fn bitsize(self) -> UInt64:
        var total: UInt64 = 0
        for i in range(self.num_hdu):
            total += self.hdu_maps[i].bitsize()
        return total

@value
struct HDUMap(CollectionElement):
    var hdu_type: HDUType
    var start_block: UInt64

    var num_header_entries: UInt64

    var dtype: FITSDataType
    var axis: TensorShape
    var GCOUNT: UInt64
    var PCOUNT: UInt64

    fn blocksize(self) -> UInt64:
        return self.header_blocksize()+self.data_blocksize()
    
    fn bitsize(self) -> UInt64:
        return self.header_bitsize()+self.data_bitsize()

    fn header_blocksize(self) -> UInt64:
        return math.ceildiv(int(self.header_bitsize()),config.fits_block_length)
    
    fn header_bitsize(self) -> UInt64:
        return self.num_header_entries*config.header_line_length

    fn data_blocksize(self) -> UInt64:
        return math.ceildiv(int(self.data_bitsize()), config.fits_block_length)
    
    fn data_bitsize(self) -> UInt64:
        return self.dtype.bitpix()*self.GCOUNT*(self.PCOUNT+self.axis.num_elements())

