from .config import config
from .header import Header, header_from_blocks
from .fitstypes import DataUnit
from .byteio import get_headers_start_blocks, get_header_sizes, get_data_sizes, get_FITS_blocks

struct FITSFileHandle:
    var file: FileHandle
    
    var _header_start_blocks: DynamicVector[Int]
    var _header_sizes: DynamicVector[Int]
    var _data_sizes: DynamicVector[Int]

    fn __init__(inout self, filename: String, mode: String) raises:
        self.file = FileHandle(filename, mode)
        self._header_start_blocks = get_headers_start_blocks(self.file)
        self._header_sizes = get_header_sizes(self.file, self._header_start_blocks)
        self._data_sizes = get_data_sizes(self.file, self._header_start_blocks, self._header_sizes)
    
    fn header(self, index: Int) raises -> Header:
        var headerblocks: Tensor[DType.int8]
        if index > len(self._header_start_blocks)-1:
            raise Error("FITS Header index out of bounds")
        else:
            headerblocks = get_FITS_blocks(self.file, self._header_start_blocks[index], self._header_start_blocks[index]+self._header_sizes[index])
        return header_from_blocks(headerblocks)

    fn data[T: DType = DType.float64](self, index: Int) -> DataUnit[T]: ...

    fn close(owned self):
        try:
            self.file.close()
        except:
            print("Trying to close a FileHandle, this shouldn't throw an error?")

    fn __del__(owned self):
        try:
            self.file.close()
        except:
            print("Trying to close a FileHandle, this shouldn't throw an error?")