struct sizes:
    alias line_length = 80
    alias keyword_length = 8
    alias valueind_length = 2
    alias field_length = 70

    alias FITSblock_size = 2880

struct keywords:
    alias SIMPLE        = "SIMPLE  "
    alias XTENSION      = "XTENSION"
    alias BITPIX        = "BITPIX  "
    alias NAXIS         = "NAXIS   "
    alias GCOUNT        = "GCOUNT  "
    alias PCOUNT        = "PCOUNT  "
    alias END           = "END     "
    alias EMPTY         = "        "

    alias SIMPLE_SIMD   = Self._to_simd(keywords.SIMPLE)
    alias XTENSION_SIMD = Self._to_simd(keywords.XTENSION)
    alias BITPIX_SIMD   = Self._to_simd(keywords.BITPIX)
    alias NAXIS_SIMD    = Self._to_simd(keywords.NAXIS)
    alias GCOUNT_SIMD   = Self._to_simd(keywords.GCOUNT)
    alias PCOUNT_SIMD   = Self._to_simd(keywords.PCOUNT)
    alias END_SIMD      = Self._to_simd(keywords.END)
    alias EMPTY_SIMD    = Self._to_simd(keywords.EMPTY)
    
    @staticmethod
    fn _to_simd(kw: StringLiteral) -> SIMD[DType.int8, 8]:
        return StringRef(kw).data.simd_load[8]()