// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <array>
#include <cmath>
#include <iostream>
#include <vector>

#include "eltwise/eltwise-add-mod-avx512.hpp"
#include "eltwise/eltwise-add-mod-internal.hpp"
#include "eltwise/eltwise-fma-mod-avx512.hpp"
#include "eltwise/eltwise-fma-mod-internal.hpp"
#include "eltwise/eltwise-mult-mod-avx512.hpp"
#include "eltwise/eltwise-mult-mod-internal.hpp"
#include "hexl/eltwise/eltwise-add-mod.hpp"
#include "hexl/eltwise/eltwise-fma-mod.hpp"
#include "hexl/eltwise/eltwise-mult-mod.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/aligned-allocator.hpp"
#include "util/util-internal.hpp"

namespace intel {
namespace hexl {

//=================================================================

static void BM_EltwiseFMAModAddNative(benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  uint64_t modulus = 0xffffffffffc0001ULL;
  bool add = state.range(1);

  auto input1 = GenerateInsecureUniformRandomValues(input_size, 0, modulus);
  uint64_t input2 = GenerateInsecureUniformRandomValue(0, modulus);
  AlignedVector64<uint64_t> input3 =
      GenerateInsecureUniformRandomValues(input_size, 0, modulus);
  uint64_t* arg3 = add ? input3.data() : nullptr;

  for (auto _ : state) {
    EltwiseFMAMod(input1.data(), input1.data(), input2, arg3, input1.size(),
                  modulus, 1);
  }
}

BENCHMARK(BM_EltwiseFMAModAddNative)
    ->Unit(benchmark::kMicrosecond)
    ->ArgsProduct({{1024, 4096, 16384}, {false, true}});

//=================================================================

static void BM_EltwiseVectorVectorFMAModAddNative(
    benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  uint64_t modulus = 0xffffffffffc0001ULL;

  auto input1 = GenerateInsecureUniformRandomValues(input_size, 0, modulus);
  auto input2 = GenerateInsecureUniformRandomValues(input_size, 0, modulus);
  auto input3 = GenerateInsecureUniformRandomValues(input_size, 0, modulus);

  for (auto _ : state) {
    EltwiseFMAMod(input1.data(), input1.data(), input2.data(), input3.data(),
                  input1.size(), modulus, 1);
  }
}

BENCHMARK(BM_EltwiseVectorVectorFMAModAddNative)
    ->Unit(benchmark::kMicrosecond)
    ->ArgsProduct({{1024, 4096, 16384}, {1}});

//=================================================================

#ifdef HEXL_HAS_AVX512DQ
static void BM_EltwiseFMAModAVX512DQ(benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  size_t modulus = 100;
  bool add = state.range(1);

  auto input1 = GenerateInsecureUniformRandomValues(input_size, 0, modulus);
  uint64_t input2 = GenerateInsecureUniformRandomValue(0, modulus);
  AlignedVector64<uint64_t> input3 =
      GenerateInsecureUniformRandomValues(input_size, 0, modulus);

  uint64_t* arg3 = add ? input3.data() : nullptr;

  for (auto _ : state) {
    EltwiseFMAModAVX512<64, 1>(input1.data(), input1.data(), input2, arg3,
                               input_size, modulus);
  }
}

BENCHMARK(BM_EltwiseFMAModAVX512DQ)
    ->Unit(benchmark::kMicrosecond)
    ->ArgsProduct({{1024, 4096, 16384}, {false, true}});
#endif

//=================================================================

#ifdef HEXL_HAS_AVX512DQ
static void BM_EltwiseVectorVectorFMAModAVX512(
    benchmark::State& state) {         //  NOLINT
  size_t input_size = state.range(0);  // N: polynomial degree
  size_t level = state.range(1);       // l: level, # of RNS moduli
  // Generate 60-bit l primes for RNS moduli
  std::vector<uint64_t> moduli = GeneratePrimes(level, 60, false, input_size);

  /* Generate random vectors which have N * l elements */
  auto reserved_size = std::pow(2, std::ceil(std::log2l(input_size * level)));
  AlignedVector64<uint64_t> input1(reserved_size), input2(reserved_size),
      input3(reserved_size);
  for (auto&& modulus : moduli) {
    AlignedVector64<uint64_t> tmp1 =
        GenerateInsecureUniformRandomValues(input_size, 0, modulus);
    AlignedVector64<uint64_t> tmp2 =
        GenerateInsecureUniformRandomValues(input_size, 0, modulus);
    AlignedVector64<uint64_t> tmp3 =
        GenerateInsecureUniformRandomValues(input_size, 0, modulus);
    input1.insert(input1.end(), tmp1.begin(), tmp1.end());
    input2.insert(input2.end(), tmp2.begin(), tmp2.end());
    input3.insert(input3.end(), tmp3.begin(), tmp3.end());
  }

  for (auto _ : state) {  // BEGIN of benchmark block
    auto input1_ptr = reinterpret_cast<uint64_t*>(input1.data());
    auto input2_ptr = reinterpret_cast<uint64_t*>(input2.data());
    auto input3_ptr = reinterpret_cast<uint64_t*>(input3.data());
    for (size_t i = 0; i < level; ++i) {
      // Perform vector-to-vector FMA
      EltwiseFMAModAVX512<64, 1>(input1_ptr, input1_ptr, input2_ptr, input3_ptr,
                                 input_size, moduli[i]);
      input1_ptr += input_size;
      input2_ptr += input_size;
      input3_ptr += input_size;
    }
  }  // END of benchmark block
}

BENCHMARK(BM_EltwiseVectorVectorFMAModAVX512)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(10000)                                      // # of iterations
    ->ArgsProduct({{static_cast<int64_t>(std::pow(2, 10)),   // logN = 10
                    static_cast<int64_t>(std::pow(2, 11)),   // logN = 11
                    static_cast<int64_t>(std::pow(2, 12)),   // logN = 12
                    static_cast<int64_t>(std::pow(2, 13)),   // logN = 13
                    static_cast<int64_t>(std::pow(2, 14)),   // logN = 14
                    static_cast<int64_t>(std::pow(2, 15))},  // logN = 15
                   {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                    15}});  // level, from 1 to 15
#endif

#ifdef HEXL_HAS_AVX512DQ
static void BM_EltwiseVectorVectorMultAddModAVX512(
    benchmark::State& state) {         //  NOLINT
  size_t input_size = state.range(0);  // N: polynomial degree
  size_t level = state.range(1);       // l: level, # of RNS moduli
  // Generate 60-bit l primes for RNS moduli
  std::vector<uint64_t> moduli = GeneratePrimes(level, 60, false, input_size);

  /* Generate random vectors which have N * l elements */
  auto reserved_size = std::pow(2, std::ceil(std::log2l(input_size * level)));
  AlignedVector64<uint64_t> input1(reserved_size), input2(reserved_size),
      input3(reserved_size);
  for (auto&& modulus : moduli) {
    AlignedVector64<uint64_t> tmp1 =
        GenerateInsecureUniformRandomValues(input_size, 0, modulus);
    AlignedVector64<uint64_t> tmp2 =
        GenerateInsecureUniformRandomValues(input_size, 0, modulus);
    AlignedVector64<uint64_t> tmp3 =
        GenerateInsecureUniformRandomValues(input_size, 0, modulus);
    input1.insert(input1.end(), tmp1.begin(), tmp1.end());
    input2.insert(input2.end(), tmp2.begin(), tmp2.end());
    input3.insert(input3.end(), tmp3.begin(), tmp3.end());
  }

  for (auto _ : state) {  // BEGIN of benchmark block
    auto input1_ptr = reinterpret_cast<uint64_t*>(input1.data());
    auto input2_ptr = reinterpret_cast<uint64_t*>(input2.data());
    auto input3_ptr = reinterpret_cast<uint64_t*>(input3.data());
    for (size_t i = 0; i < level; ++i) {
      // Perform vector-to-vector Mult
      EltwiseMultMod(input1_ptr, input1_ptr, input2_ptr, input_size, moduli[i],
                     1);
      // Perform vector-to-vector Add
      EltwiseAddMod(input1_ptr, input1_ptr, input3_ptr, input_size, moduli[i]);
      input1_ptr += input_size;
      input2_ptr += input_size;
      input3_ptr += input_size;
    }
  }  // END of benchmark block
}

BENCHMARK(BM_EltwiseVectorVectorMultAddModAVX512)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(10000)                                      // # of iterations
    ->ArgsProduct({{static_cast<int64_t>(std::pow(2, 10)),   // logN = 10
                    static_cast<int64_t>(std::pow(2, 11)),   // logN = 11
                    static_cast<int64_t>(std::pow(2, 12)),   // logN = 12
                    static_cast<int64_t>(std::pow(2, 13)),   // logN = 13
                    static_cast<int64_t>(std::pow(2, 14)),   // logN = 14
                    static_cast<int64_t>(std::pow(2, 15))},  // logN = 15
                   {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                    15}});  // level, from 1 to 15
#endif

//=================================================================

#ifdef HEXL_HAS_AVX512IFMA
static void BM_EltwiseFMAModAVX512IFMA(benchmark::State& state) {  //  NOLINT
  size_t input_size = state.range(0);
  size_t modulus = 100;
  bool add = state.range(1);

  auto input1 = GenerateInsecureUniformRandomValues(input_size, 0, modulus);
  uint64_t input2 = GenerateInsecureUniformRandomValue(0, modulus);
  auto input3 = GenerateInsecureUniformRandomValues(input_size, 0, modulus);

  uint64_t* arg3 = add ? input3.data() : nullptr;

  for (auto _ : state) {
    EltwiseFMAModAVX512<52, 1>(input1.data(), input1.data(), input2, arg3,
                               input_size, modulus);
  }
}

BENCHMARK(BM_EltwiseFMAModAVX512IFMA)
    ->Unit(benchmark::kMicrosecond)
    ->ArgsProduct({{1024, 4096, 16384}, {false, true}});
#endif

}  // namespace hexl
}  // namespace intel
