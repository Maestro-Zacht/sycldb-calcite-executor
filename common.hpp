#pragma once

#define PERFORMANCE_MEASUREMENT_ACTIVE 0
#define PERFORMANCE_REPETITIONS 100
#define USE_FUSION 1

#define SIZE_TEMP_MEMORY_GPU (((uint64_t)2) << 30) // 2GB
#define SIZE_TEMP_MEMORY_CPU (((uint64_t)4) << 30) // 4GB

#define DATA_DIR "/home/matteo/ssb/s20_columnar/"

#define SEGMENT_SIZE (((uint64_t)1) << 25)