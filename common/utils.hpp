#pragma once
#include <cstdint>
#include <random>
#include <sys/time.h>

inline uint64_t current_time_ns() __attribute__((always_inline));
inline uint64_t current_time_ns()
{
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return (t.tv_sec) * uint64_t(1000000000) + t.tv_nsec;
}

inline float next_float() __attribute__((always_inline));
inline float next_float()
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> dist(36.5, 37.5);

    return dist(gen);
}

inline void random_fill(float * ptr, int n) __attribute__((always_inline));
inline void random_fill(float * ptr, int n)
{
    for (int i = 0; i < n; ++i) {
        ptr[i] = next_float();
    }
}