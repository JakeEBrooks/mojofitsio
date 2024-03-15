from tensor import TensorShape

from mojofitsio.configs import sizes, keywords
from mojofitsio.fitstypes import FITSDType, FITSType, FITSData
from mojofitsio.header import Header
from mojofitsio.headerio import (
    get_line,
    get_field,
    get_field_nocomment,
    get_keyword,
    get_valueind,
    get_keywordSIMD,
    get_valueindSIMD
)

fn fitsopen(file: String, mode: String) raises -> FITSFileHandle:
    return FITSFileHandle(file, mode)

struct FITSFileHandle:
    var fh: FileHandle
    var map: FITSMap

    fn __init__(inout self, file: String, mode: String) raises:
        self.fh = FileHandle(file, mode)
        self.map = mapFITS(self)
    
    fn __moveinit__(inout self, owned existing: Self):
        self.fh = existing.fh^
        self.map = existing.map^

    fn __enter__(owned self) -> Self:
        return self^
    
    fn __del__(owned self):
        self.fh^.__del__()

    fn readblocks(self, startblock: UInt64) raises -> Tensor[DType.int8]:
        _ = self.fh.seek(startblock*sizes.FITSblock_size)
        return self.fh.read_bytes(sizes.FITSblock_size)
    
    fn readblocks(self, startblock: UInt64, endblock: UInt64) raises -> Tensor[DType.int8]:
        _ = self.fh.seek(startblock*sizes.FITSblock_size)
        return self.fh.read_bytes(int(endblock - startblock)*sizes.FITSblock_size)
    
    fn read_bytes(self) raises -> Tensor[DType.int8]:
        _ = self.fh.seek(0)
        return self.fh.read_bytes()
    
    fn read_bytes(self: Self, size: UInt64, offset: UInt64 = 0) raises -> Tensor[DType.int8]:
        _ = self.fh.seek(offset)
        return self.fh.read_bytes(int(size))
    
    fn read(self) raises -> String:
        _ = self.fh.seek(0)
        return self.fh.read()
    
    fn read(self: Self, size: UInt64, offset: UInt64 = 0) raises -> String:
        _ = self.fh.seek(offset)
        return self.fh.read(int(size))

    fn getHeader(self, index: Int) raises -> Header:
        return self.getHeader(self.map[index])

    fn getData[dtype: FITSDType](self, index: Int) -> FITSData[dtype]: ...

    fn getHeader(self, map: FITSMap, index: Int) raises -> Header:
        return self.getHeader(map[index])

    fn getData[dtype: FITSDType](self, map: FITSMap, index: Int) -> FITSData[dtype]: ...

    fn getHeader(self, map: HDUMap) raises -> Header:
        var outhdr = Header()
        var headerblocks = self.readblocks(map.startblock, map.startblock+map.header_blocksize())
        for line in range(map.num_headerlines):
            outhdr[get_keyword(headerblocks, line).strip()] = get_field(headerblocks, line)
        return outhdr
        

    fn getData[dtype: FITSDType](self, map: HDUMap) -> FITSData[dtype]: ...

struct FITSMap(Sized):
    var hdu_maps: DynamicVector[HDUMap]
    var num_hdu: Int

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
    var fitstype: FITSType
    var startblock: UInt64
    var num_headerlines: UInt64 # includes END
    var dtype: FITSDType
    var axis: TensorShape
    var GCOUNT: UInt64
    var PCOUNT: UInt64

    fn __str__(self) -> String:
        var outstr: String = ""
        outstr += "HDU Type: "+str(self.fitstype)+"\n"
        outstr += "Starting Block: "+str(self.startblock)+"\n"
        outstr += "Number of Header Entries: "+str(self.num_headerlines)+"\n"
        outstr += "Underlying Datatype: "+str(self.dtype)+"\n"
        outstr += "Data Axis: "+str(self.axis)
        return outstr

    fn blocksize(self) -> UInt64:
        return self.header_blocksize()+self.data_blocksize()
    
    fn bytesize(self) -> UInt64:
        return self.header_bytesize()+self.data_bytesize()

    fn header_blocksize(self) -> UInt64:
        return math.ceildiv(int(self.header_bytesize()),sizes.FITSblock_size)
    
    fn header_bytesize(self) -> UInt64:
        return self.num_headerlines*sizes.line_length

    fn data_blocksize(self) -> UInt64:
        return math.ceildiv(int(self.data_bytesize()), sizes.FITSblock_size)
    
    fn data_bytesize(self) -> UInt64:
        return self.dtype.sizeof()*self.GCOUNT*(self.PCOUNT+self.axis.num_elements())

fn mapFITS(file: FITSFileHandle) raises -> FITSMap:
    var outmap = FITSMap()
    var foundEOF = False
    var blocksize: UInt64

    outmap.append(mapHDU(file, 0)) # FITS files always have at least one HDU at the very start
    while not foundEOF:
        blocksize = outmap.blocksize()
        _ = file.fh.seek(blocksize*sizes.FITSblock_size)
        if file.fh.read() == "":
            foundEOF = True
        else:
            outmap.append(mapHDU(file, blocksize))

    return outmap

fn mapHDU(file: FITSFileHandle, startblock: UInt64) raises -> HDUMap:
    var fitstype: FITSType = FITSType.IMAGE
    var num_headerlines = _get_num_header_lines(file, startblock)
    var dtype: FITSDType = FITSDType.uint8
    var axis: TensorShape = TensorShape(0)
    var GCOUNT: UInt64 = 1
    var PCOUNT: UInt64 = 0

    var header_blocks = file.readblocks(startblock, math.ceildiv(int(num_headerlines), sizes.FITSblock_size//sizes.line_length))
    var header_line_counter: UInt64 = 0
    var kw: SIMD[DType.int8, 8]
    var field: String

    while header_line_counter < num_headerlines:
        kw = get_keywordSIMD(header_blocks, header_line_counter)
        if kw == keywords.SIMPLE_SIMD: # This is the primary HDU, therefore it is an image
            fitstype = FITSType.IMAGE
        elif kw == keywords.XTENSION_SIMD:
            field = get_field_nocomment(header_blocks, header_line_counter)
            fitstype = FITSType.from_headerfield(field.strip())
        elif kw == keywords.BITPIX_SIMD:
            field = get_field_nocomment(header_blocks, header_line_counter)
            dtype = FITSDType.from_bitpix(field.strip())
        elif kw == keywords.NAXIS_SIMD:
            field = get_field_nocomment(header_blocks, header_line_counter)
            header_line_counter += 1
            axis = _get_naxis(header_blocks, atol(field.strip()), header_line_counter)
        elif kw == keywords.GCOUNT_SIMD:
            field = get_field_nocomment(header_blocks, header_line_counter)
            GCOUNT = atol(field.strip())
        elif kw == keywords.PCOUNT_SIMD:
            field = get_field_nocomment(header_blocks, header_line_counter)
            PCOUNT = atol(field.strip())
        header_line_counter += 1
    return HDUMap(fitstype, startblock, num_headerlines, dtype, axis, GCOUNT, PCOUNT)

fn _get_num_header_lines(file: FITSFileHandle, startblock: UInt64) raises -> UInt64:
    var foundEND = False
    var block: Tensor[DType.int8]
    var blockcounter: UInt64 = 0
    var kw: SIMD[DType.int8, 8]
    var num_headerlines: UInt64 = 0
    while not foundEND:
        block = file.readblocks(startblock+blockcounter)
        for line in range(sizes.FITSblock_size//sizes.line_length):
            kw = block.data().simd_load[8](line*sizes.line_length)
            if kw == keywords.END_SIMD:
                foundEND = True
                num_headerlines += 1
            else:
                num_headerlines += 1
        blockcounter += 1
    return num_headerlines

fn _get_naxis(blocks: Tensor[DType.int8], NAXIS: Int, inout header_line_counter: UInt64) raises -> TensorShape:
    var axis = DynamicVector[Int]()
    if NAXIS == 0:
        axis.append(0)
    for i in range(NAXIS):
        axis.append(atol(get_field_nocomment(blocks, header_line_counter).strip()))
        header_line_counter += 1
    return TensorShape(axis)
