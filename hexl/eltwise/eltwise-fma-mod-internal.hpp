// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hexl/number-theory/number-theory.hpp"

namespace intel {
namespace hexl {

template <int InputModFactor>
void EltwiseFMAModNative(uint64_t* result, const uint64_t* arg1,
                         const uint64_t* arg2, const uint64_t* arg3, uint64_t n,
                         uint64_t modulus) {
  HEXL_CHECK(InputModFactor == 1 || InputModFactor == 2 || InputModFactor == 4,
             "Require InputModFactor = 1, 2, or 4")
  HEXL_CHECK(result != nullptr, "Require result != nullptr");
  HEXL_CHECK(arg1 != nullptr, "Require arg1 != nullptr");
  HEXL_CHECK(arg2 != nullptr, "Require arg2 != nullptr");
  HEXL_CHECK(n != 0, "Require n != 0");
  HEXL_CHECK(modulus > 1, "Require modulus > 1");
  HEXL_CHECK(modulus < (1ULL << 62), "Require modulus < (1ULL << 62)");
  HEXL_CHECK_BOUNDS(arg1, n, InputModFactor * modulus,
                    "arg1 exceeds bound " << (InputModFactor * modulus));
  HEXL_CHECK_BOUNDS(arg2, n, InputModFactor * modulus,
                    "arg2 exceeds bound " << (InputModFactor * modulus));

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

  const uint64_t twice_modulus = 2 * modulus;

  HEXL_LOOP_UNROLL_4
  for (size_t i = 0; i < n; ++i) {
    uint64_t prod_hi, prod_lo, c2_hi, c2_lo, Z;

    uint64_t x = ReduceMod<InputModFactor>(*arg1, modulus, &twice_modulus);
    uint64_t y = ReduceMod<InputModFactor>(*arg2, modulus, &twice_modulus);

    // Multiply inputs
    MultiplyUInt64(x, y, &prod_hi, &prod_lo);

    // floor(U / 2^{n + beta})
    uint64_t c1 = (prod_lo >> (prod_right_shift)) +
                  (prod_hi << (64 - (prod_right_shift)));

    // c2 = floor(U / 2^{n + beta}) * mu
    MultiplyUInt64(c1, barr_lo, &c2_hi, &c2_lo);

    // alpha - beta == 64, so we only need high 64 bits
    uint64_t q_hat = c2_hi;

    // only compute low bits, since we know high bits will be 0
    Z = prod_lo - q_hat * modulus;

    // Conditional subtraction
    *result = (Z >= modulus) ? (Z - modulus) : Z;

    // ADD
    if (arg3) {
      uint64_t sum = *result + *arg3;
      if (sum >= modulus) {
        *result = sum - modulus;
      } else {
        *result = sum;
      }

      ++arg3;
    }

    ++arg1;
    ++arg2;
    ++result;
  }
}

template <int InputModFactor>
void EltwiseFMAModNative(uint64_t* result, const uint64_t* arg1, uint64_t arg2,
                         const uint64_t* arg3, uint64_t n, uint64_t modulus) {
  uint64_t twice_modulus = 2 * modulus;
  uint64_t four_times_modulus = 4 * modulus;
  arg2 = ReduceMod<InputModFactor>(arg2, modulus, &twice_modulus,
                                   &four_times_modulus);

  MultiplyFactor mf(arg2, 64, modulus);
  if (arg3) {
    for (size_t i = 0; i < n; ++i) {
      uint64_t arg1_val = ReduceMod<InputModFactor>(
          *arg1++, modulus, &twice_modulus, &four_times_modulus);
      uint64_t arg3_val = ReduceMod<InputModFactor>(
          *arg3++, modulus, &twice_modulus, &four_times_modulus);

      uint64_t result_val =
          MultiplyMod(arg1_val, arg2, mf.BarrettFactor(), modulus);
      *result = AddUIntMod(result_val, arg3_val, modulus);
      result++;
    }
  } else {  // arg3 == nullptr
    for (size_t i = 0; i < n; ++i) {
      uint64_t arg1_val = ReduceMod<InputModFactor>(
          *arg1++, modulus, &twice_modulus, &four_times_modulus);
      *result++ = MultiplyMod(arg1_val, arg2, mf.BarrettFactor(), modulus);
    }
  }
}

}  // namespace hexl
}  // namespace intel
