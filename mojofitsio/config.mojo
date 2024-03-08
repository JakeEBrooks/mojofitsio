struct config:
    # Header line structure:
    # |    Keyword    |    Value Indicator   |     Value     |
    # | -- 8 bytes -- | --    2 bytes     -- | -- 70 bytes --|

    alias header_keyword_length = 8 # bytes
    alias header_field_length = 70 # bytes
    alias header_valueind_length = 2 # bytes
    alias header_line_length = 80 # bytes
    alias fits_block_length = 2880 # bytes
    
    alias END_line_str = "END     " # 8 byte keyword entry containing END in bytes 0-2
    alias END_line_simd = SIMD[DType.int8, 8](69, 78, 68, 32, 32, 32, 32, 32) # END_line_str as a SIMD of bytes