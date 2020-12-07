#pragma once

#include <limits>

#define FLT_EPSILON             std::numeric_limits<float>::epsilon()

#define KERNELS_FILENAME        "membench.cl"
#define K_READER_SINGLE_NAME    "reader_single"
#define K_COMPUTE_SINGLE_NAME   "compute_single"
#define K_WRITER_SINGLE_NAME    "writer_single"
#define K_READER_RANGE_NAME     "reader_range"
#define K_COMPUTE_RANGE_NAME    "compute_range"
#define K_WRITER_RANGE_NAME     "writer_range"


enum clKernelType
{
    Task,
    NDRange
};

enum clMemoryType
{
    Buffer,
    Shared
};
