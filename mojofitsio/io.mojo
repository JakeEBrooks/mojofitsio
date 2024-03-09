from tensor import TensorShape

from mojofitsio.config import config
from mojofitsio.header import Header
from mojofitsio.fitstypes import FITSDataType, HDUType
from mojofitsio.byteio.general import get_FITS_blocks
from mojofitsio.byteio.header import (get_header_keyword,
        get_header_field,
        get_header_field_nocomment,
        get_header_line,
        get_header_valueind)

struct FITSMap(Sized):
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
    
    fn __len__(self) -> Int:
        return self.hdu_maps.__len__()
    
    fn append(inout self, to_append: HDUMap):
        self.hdu_maps.append(to_append)
        self.num_hdu += 1
    
    fn blocksize(self) -> UInt64:
        var total: UInt64 = 0
        for i in range(self.num_hdu):
            total += self.hdu_maps[i].blocksize()
        return total
    
    fn bytesize(self) -> UInt64:
        var total: UInt64 = 0
        for i in range(self.num_hdu):
            total += self.hdu_maps[i].bytesize()
        return total

@value
struct HDUMap(CollectionElement, Stringable):
    var hdu_type: HDUType
    var startblock: UInt64

    var num_header_entries: UInt64 # includes END

    var dtype: FITSDataType
    var axis: TensorShape
    var GCOUNT: UInt64
    var PCOUNT: UInt64

    fn __str__(self) -> String:
        var outstr: String = ""
        outstr += "HDU Type: "+str(self.hdu_type)+"\n"
        outstr += "Starting Block: "+str(self.startblock)+"\n"
        outstr += "Number of Header Entries: "+str(self.num_header_entries)+"\n"
        outstr += "Underlying Datatype: "+str(self.dtype)+"\n"
        outstr += "Data Axis: "+str(self.axis)
        return outstr

    fn blocksize(self) -> UInt64:
        return self.header_blocksize()+self.data_blocksize()
    
    fn bytesize(self) -> UInt64:
        return self.header_bytesize()+self.data_bytesize()

    fn header_blocksize(self) -> UInt64:
        return math.ceildiv(int(self.header_bytesize()),config.fits_block_length)
    
    fn header_bytesize(self) -> UInt64:
        return self.num_header_entries*config.header_line_length

    fn data_blocksize(self) -> UInt64:
        return math.ceildiv(int(self.data_bytesize()), config.fits_block_length)
    
    fn data_bytesize(self) -> UInt64:
        return (int(math.abs(self.dtype.bitpix()))//8)*self.GCOUNT*(self.PCOUNT+self.axis.num_elements())

fn mapFITS(file: FileHandle) raises -> FITSMap:
    var outmap = FITSMap()
    var foundEOF: Bool = False
    var testbytes: Tensor[DType.int8]
    var blocksize: UInt64
    var _bo: UInt64
    outmap.append(mapHDU(file, 0)) # FITS files always have at least one HDU at the very start

    while not foundEOF:
        blocksize = outmap.blocksize()
        _bo = file.seek(blocksize*config.fits_block_length)
        if file.read_bytes(8).num_elements() == 0: # Don't know another way to test for EOF
            foundEOF = True
        else:
            outmap.append(mapHDU(file, blocksize))

    return outmap

fn mapHDU(file: FileHandle, startblock: UInt64) raises -> HDUMap:
    var num_header_entries = get_num_header_entries(file, startblock) # mandatory

    var header_blocks = get_FITS_blocks(file, startblock, math.ceildiv(int(num_header_entries),config.fits_block_length//config.header_line_length))
    var header_line_counter: UInt64 = 0
    var kw: String
    var field: String

    var hdu_type: HDUType = HDUType.IMAGE # mandatory, assumes an image
    var dtype: FITSDataType = FITSDataType.UI8 # mandatory, assumes bytes
    var axis: TensorShape = TensorShape(0) # mandatory, assumes 1 axis of 0 length
    var GCOUNT: UInt64 = 1 # optional, assumes GCOUNT = 1
    var PCOUNT: UInt64 = 0 # optional, assumes PCOUNT = 0

    while header_line_counter < num_header_entries:
        kw = get_header_keyword(header_blocks, header_line_counter)
        field = get_header_field_nocomment(header_blocks, header_line_counter).strip()
        if kw == "SIMPLE  ": # This is the primary HDU, therefore it is an image
            hdu_type = HDUType.IMAGE
        elif kw == "XTENSION":
            hdu_type = HDUType.from_string(field)
        elif kw == "BITPIX  ":
            dtype = FITSDataType.from_bitpix(field)
        elif kw == "NAXIS   ":
            header_line_counter += 1
            axis = _get_naxis(header_blocks, atol(field), header_line_counter)
        elif kw == "GCOUNT  ":
            GCOUNT = atol(field)
        elif kw == "PCOUNT  ":
            PCOUNT = atol(field)
        header_line_counter += 1
        
    return HDUMap(hdu_type, startblock, num_header_entries, dtype, axis, GCOUNT, PCOUNT) 


fn get_num_header_entries(file: FileHandle, startblock: UInt64) raises -> UInt64:
    alias emptykw = SIMD[DType.int8, 8].splat(32)

    var foundEND: Bool = False
    var block: Tensor[DType.int8]
    var blockcounter: UInt64 = 0
    var kw: SIMD[DType.int8, 8]
    var num_header_entries: UInt64 = 0
    while not foundEND:
        block = get_FITS_blocks(file, startblock+blockcounter)
        for line in range(config.fits_block_length//config.header_line_length):
            kw = block.data().simd_load[8](line*config.header_line_length)
            if kw == config.END_kw_simd: # If END is found
                foundEND = True
                num_header_entries += 1
            elif kw == emptykw: # If the keyword entry is empty (i.e. after END)
                pass
            else:
                num_header_entries += 1
        blockcounter += 1
    return num_header_entries

fn _get_naxis(blocks: Tensor[DType.int8], NAXIS: Int, inout header_line_counter: UInt64) raises -> TensorShape:
    var axis = DynamicVector[Int]()
    for i in range(NAXIS):
        axis.append(atol(get_header_field_nocomment(blocks, header_line_counter).strip()))
        header_line_counter += 1
    return TensorShape(axis)

