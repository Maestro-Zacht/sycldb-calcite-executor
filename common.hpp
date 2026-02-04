#pragma once

#define PERFORMANCE_MEASUREMENT_ACTIVE 1
#define PERFORMANCE_REPETITIONS 100
#define USE_FUSION 0

#define SIZE_TEMP_MEMORY_GPU (((uint64_t)20) << 30) // 20GB
#define SIZE_TEMP_MEMORY_CPU (((uint64_t)20) << 30) // 20GB

#define DATA_DIR "/home/matteo/ssb/s100_columnar/"

#define SEGMENT_SIZE (((uint64_t)1) << 30)