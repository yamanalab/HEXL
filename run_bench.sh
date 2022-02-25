#!/bin/bash

time=$(date "+%Y%m%d_%H%M%S")

numactl --cpunodebind 1 --membind 1 build/benchmark/bench_hexl \
	--benchmark_report_aggregates_only=false \
	--benchmark_display_aggregates_only=false \
	--benchmark_filter='BM_EltwiseVectorVector(FMA|MultAdd)ModAVX512' \
	--benchmark_out_format=json \
	--benchmark_out=/tmp/result/vecvecfma_bench_${time}.json
