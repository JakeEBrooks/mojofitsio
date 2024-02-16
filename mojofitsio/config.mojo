struct iobounds:
    # Header structure:
    # |    Keyword    |    Value Indicator   |     Value     |
    # | -- 8 bytes -- | --    2 bytes     -- | -- 70 bytes --|

    alias header_keyword_length = 8 # bytes
    alias header_field_length = 70 # bytes
    alias header_valueind_length = 2 # bytes
    alias header_line_length = 80 # bytes
    alias fits_block_length = 2880 # bytes