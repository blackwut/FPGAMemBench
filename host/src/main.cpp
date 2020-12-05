#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <chrono>
#include <utility>
#include <stdlib.h>
#include <math.h>
#include <limits>

#include "opencl.hpp"

using namespace std;

#define AOCL_ALIGNMENT  64
#define DATA_TYPE       cl_float

#define FLT_EPSILON     std::numeric_limits<float>::epsilon()
#define ITERATIONS      128
const vector<cl_int> batch_size = {64, 256, 1024, 4096, 16384, (1 << 16)};
const size_t mem_size = batch_size.back() * sizeof(DATA_TYPE);


#define KERNELS_FILENAME        "membench.cl"
#define K_READER_SINGLE_NAME    "reader_single"
#define K_COMPUTE_SINGLE_NAME   "compute_single"
#define K_WRITER_SINGLE_NAME    "writer_single"
#define K_READER_RANGE_NAME     "reader_range"
#define K_COMPUTE_RANGE_NAME    "compute_range"
#define K_WRITER_RANGE_NAME     "writer_range"


struct OCL {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_program program;

    void init(const std::string filename) {
        platform = clPromptPlatform();
        device = clPromptDevice(platform);
        context = clCreateContextFor(platform, device);
        program = clCreateBuildProgramFromBinary(context, device, filename);
    }

    cl_command_queue createCommandQueue() {
        cl_int status;
        cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
        clCheckErrorMsg(status, "Failed to create command queue");
        return queue;
    }

    cl_kernel createKernel(const char * kernel_name) {
        cl_int status;
        cl_kernel kernel = clCreateKernel(program, kernel_name, &status);
        clCheckErrorMsg(status, "Failed to create kernel");
        return kernel;
    }

    void clean() {
        if (program) clReleaseProgram(program);
        if (context) clReleaseContext(context);
    }
};

template <typename T>
struct clBufferShared {
    cl_context context;
    cl_command_queue queue;
    size_t size;
    cl_mem_flags buffer_flags;

    cl_mem buffer;
    T * ptr;

    clBufferShared(cl_context context,
                   cl_command_queue queue,
                   size_t size,
                   cl_mem_flags buffer_flags)
    : context(context)
    , queue(queue)
    , size(size)
    , buffer_flags(buffer_flags)
    {}

    void map(cl_map_flags flags,
             cl_event * event = NULL,
             bool blocking = true)
    {
        cl_int status;
        buffer = clCreateBuffer(context,
                                CL_MEM_ALLOC_HOST_PTR | buffer_flags,
                                size,
                                NULL, &status);
        clCheckErrorMsg(status, "Failed to create clBufferShared");
        ptr = (T *)clEnqueueMapBuffer(queue, buffer,
                                      blocking, flags,
                                      0, size,
                                      0, NULL,
                                      event, &status);
        clCheckErrorMsg(status, "Failed to map clBufferShared");
    }

    void free()
    {
        if (ptr && buffer) clEnqueueUnmapMemObject(queue, buffer, ptr, 0, NULL, NULL);
        if (buffer) clReleaseMemObject(buffer);
    }
};

// template <typename T>
// struct clBuffer {
//     cl_context context;
//     cl_command_queue queue;
//     cl_mem buffer;
//     T * ptr;
//     size_t size;

//     clBuffer(cl_context context,
//              cl_command_queue queue,
//              size_t size)
//     : context(context)
//     , queue(queue)
//     , size(size)
//     {}

//     void init(cl_mem_flags flags, cl_event * event = NULL) {
//         cl_int status;
//         buffer = clCreateBuffer(context, flags, size, NULL, &status);
//         clCheckErrorMsg(status, "Failed to create clBuffer");

//         status = posix_memalign(&ptr, AOCL_ALIGNMENT, size);
//         if (status != 0) clCheckErrorMsg(-255, "Failed to create host buffer");
//     }

//     void free() {
//         if (buffer) clReleaseMemObject(buffer);
//         if (ptr) free(ptr);
//     }
// };

float next_float()
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> dist(36.5, 37.5);

    return dist(gen);
}


void random_fill(float * ptr, int n)
{
    for (int i = 0; i < n; ++i) {
        ptr[i] = next_float();
    }
}

void check_computation(const DATA_TYPE * src, const DATA_TYPE * dst, int n)
{
    for (int i = 0; i < n; ++i) {
        const DATA_TYPE v = src[i] * src[i];
        if (fabsf(dst[i] - v) > FLT_EPSILON) {
            cerr << "ERROR: " << v << " != " << dst[i] << endl;
            exit(-2);
        }
    }
}

double get_time_events(const cl_event * events, int n) {
    double avg = 0.0;
    for (int i = 0; i < n; ++i) {
        avg += clTimeEventMS(events[i]);
    }
    return avg;
}


void test_single(OCL & ocl, cl_int n)
{
    cl_int argi;
    const size_t gws[3] = {1, 1, 1};
    const size_t lws[3] = {1, 1, 1};

    cl_command_queue queues[3];
    queues[0] = ocl.createCommandQueue();
    queues[1] = ocl.createCommandQueue();
    queues[2] = ocl.createCommandQueue();

    clBufferShared<float> src(ocl.context, queues[0],
                              mem_size, CL_MEM_READ_ONLY);
    clBufferShared<float> dst(ocl.context, queues[2],
                              mem_size, CL_MEM_WRITE_ONLY);

    cl_event event_map[2];
    src.map(CL_MAP_WRITE, &event_map[0]);
    dst.map(CL_MAP_READ, &event_map[1]);


    cout << "src.map(): " << clTimeEventMS(event_map[0]) << "\n"
         << "dst.map(): " << clTimeEventMS(event_map[1]) << "\n";


    cl_kernel kernels[3];
    kernels[0] = ocl.createKernel(K_READER_SINGLE_NAME);
    kernels[1] = ocl.createKernel(K_COMPUTE_SINGLE_NAME);
    kernels[2] = ocl.createKernel(K_WRITER_SINGLE_NAME);

    argi = 0;
    clCheckError(clSetKernelArg(kernels[0], argi++, sizeof(src.buffer), &src.buffer));
    clCheckError(clSetKernelArg(kernels[0], argi++, sizeof(n), &n));
    argi = 0;
    clCheckError(clSetKernelArg(kernels[1], argi++, sizeof(n), &n));
    argi = 0;
    clCheckError(clSetKernelArg(kernels[2], argi++, sizeof(dst.buffer), &dst.buffer));
    clCheckError(clSetKernelArg(kernels[2], argi++, sizeof(n), &n));


    cl_event event_reader[ITERATIONS];
    cl_event event_compute[ITERATIONS];
    cl_event event_writer[ITERATIONS];

    for (int i = 0; i < ITERATIONS; ++i) {
        random_fill(src.ptr, n);

        clCheckError(clEnqueueNDRangeKernel(queues[0], kernels[0],
                                            1, NULL, gws, lws,
                                            0, NULL, &event_reader[i]));
        clCheckError(clEnqueueNDRangeKernel(queues[1], kernels[1],
                                            1, NULL, gws, lws,
                                            0, NULL, &event_compute[i]));
        clCheckError(clEnqueueNDRangeKernel(queues[2], kernels[2],
                                            1, NULL, gws, lws,
                                            0, NULL, &event_writer[i]));

        clFinish(queues[2]);
        check_computation(src.ptr, dst.ptr, n);
    }

    double t_reader  = get_time_events(event_reader, ITERATIONS);
    double t_compute = get_time_events(event_compute, ITERATIONS);
    double t_writer  = get_time_events(event_writer, ITERATIONS);

    double tavg_reader  = t_reader  / ITERATIONS;
    double tavg_compute = t_compute / ITERATIONS;
    double tavg_writer  = t_writer  / ITERATIONS;

    size_t total_bytes = ITERATIONS * n * sizeof(DATA_TYPE);
    double bw_reader  = (total_bytes / 1.0e6) / t_reader;
    double bw_compute = (total_bytes * 2 / 1.0e6) / t_compute;
    double bw_writer  = (total_bytes / 1.0e6) / t_writer;

    cout << "   Iterations: " << ITERATIONS << "\n"
         << "   Batch size: " << n << "\n"
         << "  Total items: " << n * ITERATIONS << "\n"
         << " Total Memory: " << (total_bytes * 2) / (1 << 20) << " MB\n"
         << "\n";

    cout << right << fixed  << setprecision(4)
         << "┌──────────────────┬──────────┬──────────┬──────────┐\n"
         << "│                  │  reader  │ compute  │  writer  │\n"
         << "├──────────────────┼──────────┼──────────┼──────────┤\n"
         << "│       Total Time │ " << setw(8) << t_reader  << " │ " << setw(8) << t_compute  << " │ " << setw(8) << t_writer  << " │\n"
         << "│         Avg Time │ " << setw(8) << tavg_reader  << " │ " << setw(8) << tavg_compute  << " │ " << setw(8) << tavg_writer  << " │\n"
         << "│ Bandwidth (GB/s) │ " << setw(8) << bw_reader << " │ " << setw(8) << bw_compute << " │ " << setw(8) << bw_writer << " │\n"
         << "└──────────────────┴──────────┴──────────┴──────────┘\n";

    src.free();
    dst.free();

    for (int i = 0; i < 3; ++i) if (kernels[i]) clReleaseKernel(kernels[i]);
    for (int i = 0; i < 3; ++i) if (queues[i]) clReleaseCommandQueue(queues[i]);
}

int main(int argc, char * argv[])
{
    OCL ocl;
    ocl.init(argv[1]);

    cout << mem_size << endl;
    test_single(ocl, batch_size.back());

    // TODO: warmup buffers

    // // clEnequeueWriteBuffer()
    // for (const auto bs : batch_size) {

    // }

    // // clEnqueueMapBuffer()
    // for (const auto bs : batch_size) {

    // }

    // double time_execution = 1.e-9 * (end_execution - start_execution);
    // double time_fastflow = pipe.ffTime();
    // double throughput = (stream_size * batch_size) / time_execution;
    
    // std::cout.precision(4);
    // std::cout << "    Time Execution: " << std::fixed << time_execution << " s\n";
    // std::cout << "          FastFlow: " << std::fixed << time_fastflow << " ms\n";
    // std::cout << "        Throughput: " << (int)throughput << " tuples/second\n";
    // std::cout << "\n";
    // std::cout << "********* Kernels *********\n";
    // std::cout << "             write: " << std::fixed << getTotalTimeEvents(events_write)     << " ms\n";
    // std::cout << "         generator: " << std::fixed << getTotalTimeEvents(events_generator) << " ms\n";
    // std::cout << "              sink: " << std::fixed << getTotalTimeEvents(events_sink)      << " ms\n";
    // std::cout << "              read: " << std::fixed << getTotalTimeEvents(events_read)      << " ms\n";
    // std::cout << "average_calculator: " << std::fixed << clTimeEventMS(event_avg_calculator)  << " ms\n";
    // std::cout << "          detector: " << std::fixed << clTimeEventMS(event_detector)        << " ms\n";

    ocl.clean();

    return 0;
}