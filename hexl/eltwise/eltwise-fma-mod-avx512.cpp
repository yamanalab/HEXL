// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "eltwise/eltwise-fma-mod-avx512.hpp"

#include <immintrin.h>

#include "hexl/eltwise/eltwise-fma-mod.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/check.hpp"
#include "util/avx512-util.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_HAS_AVX512IFMA
template void EltwiseFMAModAVX512<52, 1>(uint64_t* result, const uint64_t* arg1,
                                         const uint64_t* arg2,
                                         const uint64_t* arg3, uint64_t n,
                                         uint64_t modulus);
// template void EltwiseFMAModAVX512<52, 2>(uint64_t* result, const uint64_t*
// arg1,
//                                          uint64_t* arg2, const uint64_t*
//                                          arg3, uint64_t n, uint64_t modulus);
// template void EltwiseFMAModAVX512<52, 4>(uint64_t* result, const uint64_t*
// arg1,
//                                          uint64_t* arg2, const uint64_t*
//                                          arg3, uint64_t n, uint64_t modulus);
// template void EltwiseFMAModAVX512<52, 8>(uint64_t* result, const uint64_t*
// arg1,
//                                          uint64_t* arg2, const uint64_t*
//                                          arg3, uint64_t n, uint64_t modulus);
template void EltwiseFMAModAVX512<52, 1>(uint64_t* result, const uint64_t* arg1,
                                         uint64_t arg2, const uint64_t* arg3,
                                         uint64_t n, uint64_t modulus);
template void EltwiseFMAModAVX512<52, 2>(uint64_t* result, const uint64_t* arg1,
                                         uint64_t arg2, const uint64_t* arg3,
                                         uint64_t n, uint64_t modulus);
template void EltwiseFMAModAVX512<52, 4>(uint64_t* result, const uint64_t* arg1,
                                         uint64_t arg2, const uint64_t* arg3,
                                         uint64_t n, uint64_t modulus);
template void EltwiseFMAModAVX512<52, 8>(uint64_t* result, const uint64_t* arg1,
                                         uint64_t arg2, const uint64_t* arg3,
                                         uint64_t n, uint64_t modulus);
#endif

#ifdef HEXL_HAS_AVX512DQ
template void EltwiseFMAModAVX512<64, 1>(uint64_t* result, const uint64_t* arg1,
                                         const uint64_t* arg2,
                                         const uint64_t* arg3, uint64_t n,
                                         uint64_t modulus);
// template void EltwiseFMAModAVX512<64, 2>(uint64_t* result, const uint64_t*
// arg1,
//                                          uint64_t* arg2, const uint64_t*
//                                          arg3, uint64_t n, uint64_t modulus);
// template void EltwiseFMAModAVX512<64, 4>(uint64_t* result, const uint64_t*
// arg1,
//                                          uint64_t* arg2, const uint64_t*
//                                          arg3, uint64_t n, uint64_t modulus);
// template void EltwiseFMAModAVX512<64, 8>(uint64_t* result, const uint64_t*
// arg1,
//                                          uint64_t* arg2, const uint64_t*
//                                          arg3, uint64_t n, uint64_t modulus);
template void EltwiseFMAModAVX512<64, 1>(uint64_t* result, const uint64_t* arg1,
                                         uint64_t arg2, const uint64_t* arg3,
                                         uint64_t n, uint64_t modulus);
template void EltwiseFMAModAVX512<64, 2>(uint64_t* result, const uint64_t* arg1,
                                         uint64_t arg2, const uint64_t* arg3,
                                         uint64_t n, uint64_t modulus);
template void EltwiseFMAModAVX512<64, 4>(uint64_t* result, const uint64_t* arg1,
                                         uint64_t arg2, const uint64_t* arg3,
                                         uint64_t n, uint64_t modulus);
template void EltwiseFMAModAVX512<64, 8>(uint64_t* result, const uint64_t* arg1,
                                         uint64_t arg2, const uint64_t* arg3,
                                         uint64_t n, uint64_t modulus);

#endif

#ifdef HEXL_HAS_AVX512DQ

template <int BitShift, int InputModFactor>
void EltwiseFMAModAVX512(uint64_t* result, const uint64_t* arg1,
                         const uint64_t* arg2, const uint64_t* arg3, uint64_t n,
                         uint64_t modulus) {
  HEXL_CHECK(BitShift == 52 || BitShift == 64,
             "Invalid bitshift " << BitShift << "; need 52 or 64");

  HEXL_CHECK(InputModFactor == 1, "Require InputModFactor = 1")
  HEXL_CHECK(modulus < MaximumValue(BitShift),
             "Modulus " << modulus << " exceeds bit shift bound "
                        << MaximumValue(BitShift));
  HEXL_CHECK(modulus != 0, "Require modulus != 0");
  HEXL_CHECK(InputModFactor * modulus < (1ULL << 63),
             "Require InputModFactor * modulus < (1ULL << 63)");
  HEXL_CHECK(modulus < (1ULL << 62), "Require  modulus < (1ULL << 62)");
  HEXL_CHECK(modulus > 1, "Require modulus > 1");

  HEXL_CHECK(arg1, "arg1 == nullptr");
  HEXL_CHECK(arg2, "arg2 == nullptr");
  HEXL_CHECK(arg3, "arg3 == nullptr");
  HEXL_CHECK(result, "result == nullptr");
  HEXL_CHECK_BOUNDS(arg1, n, InputModFactor * modulus,
                    "arg1 exceeds bound " << (InputModFactor * modulus));
  HEXL_CHECK_BOUNDS(arg2, n, InputModFactor * modulus,
                    "arg2 exceeds bound " << (InputModFactor * modulus));
  // TODO(musaprg): Is this really needed?
  HEXL_CHECK_BOUNDS(arg3, n, InputModFactor * modulus,
                    "arg3 exceeds bound " << (InputModFactor * modulus));

  // TODO(musaprg): This check is for using Float method when the modulus size
  // is under 50 bit for efficiency.
  //                For now ignore it and use Int case for all cases.
  // HEXL_CHECK(InputModFactor * modulus > (1ULL << 50),
  //            "Require InputModFactor * modulus > (1ULL << 50)")

  uint64_t n_mod_8 = n % 8;
  if (n_mod_8 != 0) {
    EltwiseFMAModNative<InputModFactor>(result, arg1, arg2, arg3, n_mod_8,
                                        modulus);
    arg1 += n_mod_8;
    arg2 += n_mod_8;
    if (arg3 != nullptr) {
      arg3 += n_mod_8;
    }
    result += n_mod_8;
    n -= n_mod_8;
  }

  // Port from EltwiseMultModAVX512DQInt
  // Algorithm 2 from
  // https://homes.esat.kuleuven.be/~fvercaut/papers/bar_mont.pdf
  constexpr int64_t beta = -2;
  HEXL_CHECK(beta <= -2, "beta must be <= -2 for correctness");
  constexpr int64_t alpha = 62;  // ensures alpha - beta = 64
  uint64_t gamma = Log2(InputModFactor);
  HEXL_UNUSED(gamma);
  HEXL_CHECK(alpha >= gamma + 1, "alpha must be >= gamma + 1 for correctness");

  const uint64_t ceil_log_mod = Log2(modulus) + 1;  // "n" from Algorithm 2
  uint64_t prod_right_shift = ceil_log_mod + beta;

  // Barrett factor "mu"
  // TODO(fboemer): Allow MultiplyFactor to take bit shifts != 64
  HEXL_CHECK(ceil_log_mod + alpha >= 64, "ceil_log_mod + alpha < 64");
  uint64_t barr_lo =
      MultiplyFactor(uint64_t(1) << (ceil_log_mod + alpha - 64), 64, modulus)
          .BarrettFactor();

  __m512i v_barr_lo = _mm512_set1_epi64(static_cast<int64_t>(barr_lo));
  __m512i v_modulus = _mm512_set1_epi64(static_cast<int64_t>(modulus));
  __m512i v_twice_mod = _mm512_set1_epi64(static_cast<int64_t>(2 * modulus));
  __m512i v_quad_mod = _mm512_set1_epi64(static_cast<int64_t>(4 * modulus));
  const __m512i* vp_arg1 = reinterpret_cast<const __m512i*>(arg1);
  const __m512i* vp_arg2 = reinterpret_cast<const __m512i*>(arg2);
  __m512i* vp_result = reinterpret_cast<__m512i*>(result);

  HEXL_UNUSED(v_twice_mod);

  const __m512i* vp_arg3 = reinterpret_cast<const __m512i*>(arg3);
  HEXL_LOOP_UNROLL_4
  for (size_t i = n / 8; i > 0; --i) {
    // -------------------------------------------------------------
    // <MULT>-------------------------------------------------------
    __m512i v_op1 = _mm512_loadu_si512(vp_arg1);
    __m512i v_op2 = _mm512_loadu_si512(vp_arg2);

    v_op1 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op1, v_modulus,
                                                        &v_twice_mod);

    v_op2 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op2, v_modulus,
                                                        &v_twice_mod);

    __m512i v_prod_hi = _mm512_hexl_mulhi_epi<64>(v_op1, v_op2);
    __m512i v_prod_lo = _mm512_hexl_mullo_epi<64>(v_op1, v_op2);

    // c1 = floor(U / 2^{n + beta})
    __m512i c1 = _mm512_hexl_shrdi_epi64(
        v_prod_lo, v_prod_hi, static_cast<unsigned int>(prod_right_shift));

    // alpha - beta == 64, so we only need high 64 bits
    // Perform approximate computation of high bits, as described on page
    // 7 of https://arxiv.org/pdf/2003.04510.pdf
    __m512i q_hat = _mm512_hexl_mulhi_approx_epi<64>(c1, v_barr_lo);
    __m512i v_result = _mm512_hexl_mullo_epi<64>(q_hat, v_modulus);
    // Computes result in [0, 4q)
    v_result = _mm512_sub_epi64(v_prod_lo, v_result);

    // TODO(musaprg): Consider if the lazy reduction is valid
    // // Reduce result to [0, q)
    // v_result =
    //     _mm512_hexl_small_mod_epu64<4>(v_result, v_modulus, &v_twice_mod);
    // </MULT>------------------------------------------------------
    // -------------------------------------------------------------

    // -------------------------------------------------------------
    // <ADD>---------------------------------------------------------
    __m512i v_op3 = _mm512_loadu_si512(vp_arg3);
    // Computes result in [0, 5q)
    // v_result = _mm512_hexl_small_add_mod_epi64(v_result, v_op3, v_modulus);
    v_result = _mm512_add_epi64(v_result, v_op3);
    // Reduce result to [0, q)
    v_result = _mm512_hexl_small_mod_epu64<8>(v_result, v_modulus, &v_twice_mod,
                                              &v_quad_mod);
    _mm512_storeu_si512(vp_result, v_result);
    // </ADD>--------------------------------------------------------
    // -------------------------------------------------------------

    ++vp_arg1;
    ++vp_arg2;
    ++vp_arg3;
    ++vp_result;
  }

  HEXL_CHECK_BOUNDS(result, n, modulus, "result exceeds bound " << modulus);
}

/// uses Shoup's modular multiplication. See Algorithm 4 of
/// https://arxiv.org/pdf/2012.01968.pdf
template <int BitShift, int InputModFactor>
void EltwiseFMAModAVX512(uint64_t* result, const uint64_t* arg1, uint64_t arg2,
                         const uint64_t* arg3, uint64_t n, uint64_t modulus) {
  HEXL_CHECK(modulus < MaximumValue(BitShift),
             "Modulus " << modulus << " exceeds bit shift bound "
                        << MaximumValue(BitShift));
  HEXL_CHECK(modulus != 0, "Require modulus != 0");

  HEXL_CHECK(arg1, "arg1 == nullptr");
  HEXL_CHECK(result, "result == nullptr");

  HEXL_CHECK_BOUNDS(arg1, n, InputModFactor * modulus,
                    "arg1 exceeds bound " << (InputModFactor * modulus));
  HEXL_CHECK_BOUNDS(&arg2, 1, InputModFactor * modulus,
                    "arg2 exceeds bound " << (InputModFactor * modulus));
  HEXL_CHECK(BitShift == 52 || BitShift == 64,
             "Invalid bitshift " << BitShift << "; need 52 or 64");

  uint64_t n_mod_8 = n % 8;
  if (n_mod_8 != 0) {
    EltwiseFMAModNative<InputModFactor>(result, arg1, arg2, arg3, n_mod_8,
                                        modulus);
    arg1 += n_mod_8;
    if (arg3 != nullptr) {
      arg3 += n_mod_8;
    }
    result += n_mod_8;
    n -= n_mod_8;
  }

  uint64_t twice_modulus = 2 * modulus;
  uint64_t four_times_modulus = 4 * modulus;
  arg2 = ReduceMod<InputModFactor>(arg2, modulus, &twice_modulus,
                                   &four_times_modulus);
  uint64_t arg2_barr = MultiplyFactor(arg2, BitShift, modulus).BarrettFactor();

  __m512i varg2_barr = _mm512_set1_epi64(static_cast<int64_t>(arg2_barr));

  __m512i vmodulus = _mm512_set1_epi64(static_cast<int64_t>(modulus));
  __m512i vneg_modulus = _mm512_set1_epi64(-static_cast<int64_t>(modulus));
  __m512i v2_modulus = _mm512_set1_epi64(static_cast<int64_t>(2 * modulus));
  __m512i v4_modulus = _mm512_set1_epi64(static_cast<int64_t>(4 * modulus));
  const __m512i* vp_arg1 = reinterpret_cast<const __m512i*>(arg1);
  __m512i varg2 = _mm512_set1_epi64(static_cast<int64_t>(arg2));
  varg2 = _mm512_hexl_small_mod_epu64<InputModFactor>(varg2, vmodulus,
                                                      &v2_modulus, &v4_modulus);

  __m512i* vp_result = reinterpret_cast<__m512i*>(result);

  if (arg3) {
    const __m512i* vp_arg3 = reinterpret_cast<const __m512i*>(arg3);
    HEXL_LOOP_UNROLL_8
    for (size_t i = n / 8; i > 0; --i) {
      __m512i varg1 = _mm512_loadu_si512(vp_arg1);
      __m512i varg3 = _mm512_loadu_si512(vp_arg3);

      varg1 = _mm512_hexl_small_mod_epu64<InputModFactor>(
          varg1, vmodulus, &v2_modulus, &v4_modulus);
      varg3 = _mm512_hexl_small_mod_epu64<InputModFactor>(
          varg3, vmodulus, &v2_modulus, &v4_modulus);

      __m512i va_times_b = _mm512_hexl_mullo_epi<BitShift>(varg1, varg2);
      __m512i vq = _mm512_hexl_mulhi_epi<BitShift>(varg1, varg2_barr);

      // Compute vq in [0, 2 * p) where p is the modulus
      // a * b - q * p
      vq = _mm512_hexl_mullo_add_lo_epi<BitShift>(va_times_b, vq, vneg_modulus);

      // Add arg3, bringing vq to [0, 3 * p)
      vq = _mm512_add_epi64(vq, varg3);
      // Reduce to [0, p)
      vq = _mm512_hexl_small_mod_epu64<4>(vq, vmodulus, &v2_modulus);

      _mm512_storeu_si512(vp_result, vq);

      ++vp_arg1;
      ++vp_result;
      ++vp_arg3;
    }
  } else {  // arg3 == nullptr
    HEXL_LOOP_UNROLL_8
    for (size_t i = n / 8; i > 0; --i) {
      __m512i varg1 = _mm512_loadu_si512(vp_arg1);
      varg1 = _mm512_hexl_small_mod_epu64<InputModFactor>(
          varg1, vmodulus, &v2_modulus, &v4_modulus);

      __m512i va_times_b = _mm512_hexl_mullo_epi<BitShift>(varg1, varg2);
      __m512i vq = _mm512_hexl_mulhi_epi<BitShift>(varg1, varg2_barr);

      // Compute vq in [0, 2 * p) where p is the modulus
      // a * b - q * p
      vq = _mm512_hexl_mullo_add_lo_epi<BitShift>(va_times_b, vq, vneg_modulus);
      // Conditional Barrett subtraction
      vq = _mm512_hexl_small_mod_epu64(vq, vmodulus);
      _mm512_storeu_si512(vp_result, vq);

      ++vp_arg1;
      ++vp_result;
    }
  }
}

#endif

}  // namespace hexl
}  // namespace intel
