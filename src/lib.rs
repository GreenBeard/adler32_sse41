use std::arch::x86_64::__m128i;
use std::arch::x86_64::{
    _mm_add_epi32, _mm_blend_epi16, _mm_cvtepu8_epi32, _mm_extract_epi32, _mm_hadd_epi32,
    _mm_load_si128, _mm_loadu_si128, _mm_mul_epu32, _mm_mullo_epi32, _mm_set1_epi32,
    _mm_setzero_si128, _mm_slli_epi32, _mm_srli_epi32, _mm_srli_si128, _mm_sub_epi32,
};
use std::convert::TryFrom;
use std::ptr::addr_of;

#[repr(C, align(128))]
struct AlignedXmmMem {
    data: [u8; 16],
}

impl AlignedXmmMem {
    fn as_xmm_ptr(&self) -> *const __m128i {
        self as *const AlignedXmmMem as *const __m128i
    }
}

// Assumes a 4 by u32 input __m128i. Generated using
// https://www.officedaytime.com/simd512e/simdimg/constintdiv.html . Thank you
// to them <3
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn adler_modulo_65521(dividends: __m128i) -> __m128i {
    let magic: __m128i = _mm_set1_epi32(-2146992015);
    let t0: __m128i = _mm_mul_epu32(magic, dividends);
    let t1: __m128i = _mm_mul_epu32(magic, _mm_srli_si128(dividends, 4));
    let t: __m128i = _mm_blend_epi16(_mm_srli_si128(t0, 4), t1, 0xcc);
    let quotients: __m128i = _mm_srli_epi32(t, 15);
    _mm_sub_epi32(dividends, _mm_mullo_epi32(quotients, _mm_set1_epi32(65521)))
}

/// # Safety
/// Safe as long as sse4.1 is available.
#[target_feature(enable = "sse4.1")]
pub unsafe fn unaligned_simd_adler32_amd64(data: &[u8]) -> u32 {
    let chunk_count: usize = data.len() / 16;

    let mut a0: __m128i = _mm_setzero_si128();
    let mut a1: __m128i = _mm_setzero_si128();
    let mut a2: __m128i = _mm_setzero_si128();
    let mut a3: __m128i = _mm_setzero_si128();
    let mut b0: __m128i = _mm_setzero_si128();
    let mut b1: __m128i = _mm_setzero_si128();
    let mut b2: __m128i = _mm_setzero_si128();
    let mut b3: __m128i = _mm_setzero_si128();
    for i in 0..chunk_count {
        let data_chunk: __m128i = _mm_loadu_si128(addr_of!(data[16 * i]) as *const __m128i);

        let data0: __m128i = _mm_cvtepu8_epi32(data_chunk);
        let data1: __m128i = _mm_cvtepu8_epi32(_mm_srli_si128(data_chunk, 4));
        let data2: __m128i = _mm_cvtepu8_epi32(_mm_srli_si128(data_chunk, 8));
        let data3: __m128i = _mm_cvtepu8_epi32(_mm_srli_si128(data_chunk, 12));

        b0 = _mm_add_epi32(_mm_add_epi32(b0, _mm_slli_epi32(a0, 4)), data0);
        b1 = _mm_add_epi32(_mm_add_epi32(b1, _mm_slli_epi32(a1, 4)), data1);
        b2 = _mm_add_epi32(_mm_add_epi32(b2, _mm_slli_epi32(a2, 4)), data2);
        b3 = _mm_add_epi32(_mm_add_epi32(b3, _mm_slli_epi32(a3, 4)), data3);

        a0 = _mm_add_epi32(a0, data0);
        a1 = _mm_add_epi32(a1, data1);
        a2 = _mm_add_epi32(a2, data2);
        a3 = _mm_add_epi32(a3, data3);

        // Keep things from overflowing.
        if i % 1215 == 0 {
            a0 = adler_modulo_65521(a0);
            a1 = adler_modulo_65521(a1);
            a2 = adler_modulo_65521(a2);
            a3 = adler_modulo_65521(a3);
            b0 = adler_modulo_65521(b0);
            b1 = adler_modulo_65521(b1);
            b2 = adler_modulo_65521(b2);
            b3 = adler_modulo_65521(b3);
        }
    }

    a0 = adler_modulo_65521(a0);
    a1 = adler_modulo_65521(a1);
    a2 = adler_modulo_65521(a2);
    a3 = adler_modulo_65521(a3);
    b0 = adler_modulo_65521(b0);
    b1 = adler_modulo_65521(b1);
    b2 = adler_modulo_65521(b2);
    b3 = adler_modulo_65521(b3);

    static A0_MUL_TABLE: AlignedXmmMem = AlignedXmmMem {
        data: unsafe { std::mem::transmute([15u32, 14u32, 13u32, 12u32]) },
    };
    static A1_MUL_TABLE: AlignedXmmMem = AlignedXmmMem {
        data: unsafe { std::mem::transmute([11u32, 10u32, 9u32, 8u32]) },
    };
    static A2_MUL_TABLE: AlignedXmmMem = AlignedXmmMem {
        data: unsafe { std::mem::transmute([7u32, 6u32, 5u32, 4u32]) },
    };
    static A3_MUL_TABLE: AlignedXmmMem = AlignedXmmMem {
        data: unsafe { std::mem::transmute([3u32, 2u32, 1u32, 0u32]) },
    };
    let a0_mul_table: __m128i = _mm_load_si128(A0_MUL_TABLE.as_xmm_ptr());
    let a1_mul_table: __m128i = _mm_load_si128(A1_MUL_TABLE.as_xmm_ptr());
    let a2_mul_table: __m128i = _mm_load_si128(A2_MUL_TABLE.as_xmm_ptr());
    let a3_mul_table: __m128i = _mm_load_si128(A3_MUL_TABLE.as_xmm_ptr());

    b0 = _mm_add_epi32(b0, _mm_mullo_epi32(a0, a0_mul_table));
    b1 = _mm_add_epi32(b1, _mm_mullo_epi32(a1, a1_mul_table));
    b2 = _mm_add_epi32(b2, _mm_mullo_epi32(a2, a2_mul_table));
    b3 = _mm_add_epi32(b3, _mm_mullo_epi32(a3, a3_mul_table));

    let a01: __m128i = _mm_hadd_epi32(a0, a1);
    let a23: __m128i = _mm_hadd_epi32(a2, a3);
    let b01: __m128i = _mm_hadd_epi32(b0, b1);
    let b23: __m128i = _mm_hadd_epi32(b2, b3);

    let a0123: __m128i = _mm_hadd_epi32(a01, a23);
    let b0123: __m128i = _mm_hadd_epi32(b01, b23);

    let zero: __m128i = _mm_setzero_si128();
    let a: __m128i = _mm_hadd_epi32(_mm_hadd_epi32(a0123, zero), zero);
    let b: __m128i = _mm_hadd_epi32(_mm_hadd_epi32(b0123, zero), zero);

    let mut a_scalar: u32 = _mm_extract_epi32(a, 0) as u32;
    let mut b_scalar: u32 = _mm_extract_epi32(b, 0) as u32;

    a_scalar = (a_scalar + 1) % 65521;
    b_scalar = (b_scalar + u32::try_from((16 * chunk_count) % 65521).unwrap()) % 65521;

    for byte in data[(16 * chunk_count)..].iter() {
        a_scalar += u32::from(*byte);
        b_scalar += a_scalar;
    }

    (b_scalar % 65521) << 16 | (a_scalar % 65521)
}

/// A slow, but very easy to reason about implementation.
pub fn simple_adler32(data: &[u8]) -> u32 {
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for byte in data.iter() {
        a = (a + u32::from(*byte)) % 65521;
        b = (b + a) % 65521;
    }
    b << 16 | a
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_simple_adler32() {
        assert_eq!(crate::simple_adler32(b"Wikipedia"), 0x11e60398);
    }

    #[test]
    fn test_unaligned_simd_adler32_amd64() {
        if std::is_x86_feature_detected!("sse4.1") {
            let datas: &[&[u8]] = &[
                &[],
                &[
                    0xf6, 0x22, 0x88, 0xcb, 0x0a, 0xeb, 0x33, 0x06, 0xbe, 0xb5, 0xf2, 0xa9, 0x26,
                    0x46, 0x60, 0xf0, 0x97, 0x26, 0x46, 0x07, 0x43, 0xd5, 0x87, 0xe8, 0x3a, 0x70,
                    0x7c, 0xc5, 0x94, 0x48, 0x63, 0x0c, 0x06, 0xe3, 0x4a, 0xb0, 0xb7, 0x74, 0x54,
                    0x1b, 0x75, 0xed, 0xe0, 0x75, 0x6f, 0x25, 0x13, 0x96, 0xb7, 0x0b, 0x35, 0x34,
                    0xe3, 0x97, 0x1b, 0x9a, 0x8b, 0x96, 0x41, 0x8f, 0x4b, 0x55, 0xc4, 0x03, 0x94,
                    0x16, 0xef, 0x24, 0x42, 0x38, 0xe1, 0xe8, 0xf5, 0xb3, 0x50, 0x3f, 0x5f, 0x2a,
                    0x0d, 0xda, 0x8d, 0xdb, 0x35, 0x87, 0xde, 0x31, 0xb7, 0xf0, 0x5f, 0x36, 0x60,
                    0x41, 0x4b, 0x02, 0x67, 0xcc, 0x48, 0xe0, 0x73, 0x23, 0x6a, 0xf3, 0x22, 0x53,
                    0xb8, 0xf7, 0xbc, 0x29, 0xa0, 0xbb, 0x06, 0x36, 0x63, 0xfa, 0x7c, 0xe6, 0x94,
                    0x87, 0x1b, 0x8f, 0x0e, 0xbc, 0x81, 0x99, 0x66, 0x43, 0xf1, 0x59, 0x7f, 0xd1,
                    0x57, 0x1b, 0x52, 0xa6, 0x42, 0xa9, 0x51, 0x4d, 0xca, 0x58, 0xc5, 0xa8, 0x70,
                    0x3f, 0xcc,
                ],
            ];

            for &data in datas.iter() {
                let simple: u32 = crate::simple_adler32(data);
                let simd: u32 = unsafe { crate::unaligned_simd_adler32_amd64(data) };
                println!("0x{:x}", simple);
                println!("0x{:x}", simd);
                assert_eq!(simple, simd);
            }
        }
    }

    #[test]
    fn test_no_overflow_unaligned_simd_adler32_amd64() {
        if std::is_x86_feature_detected!("sse4.1") {
            let data: Vec<u8> = vec![255u8; 1024 * 1024 * 1024 + 7];

            let simple: u32 = crate::simple_adler32(data.as_slice());
            let simd: u32 = unsafe { crate::unaligned_simd_adler32_amd64(data.as_slice()) };
            println!("0x{:x}", simple);
            println!("0x{:x}", simd);
            assert_eq!(simple, simd);
        }
    }
}
