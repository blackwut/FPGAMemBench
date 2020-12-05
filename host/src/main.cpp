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
#include "options.hpp"

using namespace std;

#define AOCL_ALIGNMENT  64
#define DATA_TYPE       cl_float
#define FLT_EPSILON     std::numeric_limits<float>::epsilon()


#define KERNELS_FILENAME        "membench.cl"
#define K_READER_SINGLE_NAME    "reader_single"
#define K_COMPUTE_SINGLE_NAME   "compute_single"
#define K_WRITER_SINGLE_NAME    "writer_single"
#define K_READER_RANGE_NAME     "reader_range"
#define K_COMPUTE_RANGE_NAME    "compute_range"
#define K_WRITER_RANGE_NAME     "writer_range"


struct OCL
{
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_program program;

    void init(const std::string filename, int platformid = -1, int deviceid = -1) {
        platform = (platformid < 0) ? clPromptPlatform() : clSelectPlatform(platformid);
        device = (deviceid < 0) ? clPromptDevice(platform) : clSelectDevice(platform, deviceid);
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
        avg += clTimeEventNS(events[i]);
    }
    return avg;
}

void print_results(int iterations, int size,
                   cl_event * event_reader,
                   cl_event * event_compute,
                   cl_event * event_writer)
{
    double t_reader  = get_time_events(event_reader, iterations);
    double t_compute = get_time_events(event_compute, iterations);
    double t_writer  = get_time_events(event_writer, iterations);

    double tavg_reader  = t_reader  / iterations;
    double tavg_compute = t_compute / iterations;
    double tavg_writer  = t_writer  / iterations;

    size_t total_bytes = iterations * size * sizeof(DATA_TYPE);
    double bw_reader  = total_bytes / t_reader;
    double bw_compute = total_bytes / t_compute;
    double bw_writer  = total_bytes / t_writer;

    cout << "   Iterations: " << iterations                    << "\n"
         << "   Batch size: " << size                          << "\n"
         << "  Total items: " << iterations * size             << "\n"
         << " Total Memory: " << (total_bytes * 2) / (1 << 20) << " MB\n"
         << "\n";

    cout << right << fixed  << setprecision(4)
         << "┌──────────────────┬────────────┬────────────┬────────────┐\n"
         << "│                  │   reader   │  compute   │   writer   │\n"
         << "├──────────────────┼────────────┼────────────┼────────────┤\n"
         << "│ Total Time  (ms) │ " << setw(10) << t_reader     * 1.0e-6 << " │ "
                                    << setw(10) << t_compute    * 1.0e-6 << " │ "
                                    << setw(10) << t_writer     * 1.0e-6 << " │\n"
         << "│   Avg Time  (ms) │ " << setw(10) << tavg_reader  * 1.0e-6 << " │ "
                                    << setw(10) << tavg_compute * 1.0e-6 << " │ "
                                    << setw(10) << tavg_writer  * 1.0e-6 << " │\n"
         << "│ Bandwidth (GB/s) │ " << setw(10) << bw_reader             << " │ "
                                    << setw(10) << bw_compute            << " │ "
                                    << setw(10) << bw_writer             << " │\n"
         << "└──────────────────┴────────────┴────────────┴────────────┘\n";
}

void test_single(OCL & ocl, int iterations, int size)
{
    cl_int argi;
    const size_t gws[3] = {1, 1, 1};
    const size_t lws[3] = {1, 1, 1};

    cl_command_queue queues[3];
    queues[0] = ocl.createCommandQueue();
    queues[1] = ocl.createCommandQueue();
    queues[2] = ocl.createCommandQueue();

    clBufferShared<float> src(ocl.context, queues[0],
                              size * sizeof(DATA_TYPE), CL_MEM_READ_ONLY);
    clBufferShared<float> dst(ocl.context, queues[2],
                              size * sizeof(DATA_TYPE), CL_MEM_WRITE_ONLY);

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
    clCheckError(clSetKernelArg(kernels[0], argi++, sizeof(size), &size));
    argi = 0;
    clCheckError(clSetKernelArg(kernels[1], argi++, sizeof(size), &size));
    argi = 0;
    clCheckError(clSetKernelArg(kernels[2], argi++, sizeof(dst.buffer), &dst.buffer));
    clCheckError(clSetKernelArg(kernels[2], argi++, sizeof(size), &size));


    cl_event * event_reader  = (cl_event *)malloc(sizeof(cl_event) * iterations);
    cl_event * event_compute = (cl_event *)malloc(sizeof(cl_event) * iterations);
    cl_event * event_writer = (cl_event *)malloc(sizeof(cl_event) * iterations);

    for (int i = 0; i < iterations; ++i) {
        random_fill(src.ptr, size);

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
        check_computation(src.ptr, dst.ptr, size);
    }

    print_results(iterations, size, event_reader, event_compute, event_writer);

    free(event_reader);
    free(event_compute);
    free(event_writer);

    src.free();
    dst.free();
    for (int i = 0; i < 3; ++i) if (kernels[i]) clReleaseKernel(kernels[i]);
    for (int i = 0; i < 3; ++i) if (queues[i]) clReleaseCommandQueue(queues[i]);
}

int main(int argc, char * argv[])
{
    Options opt;
    opt.process_args(argc, argv);

    OCL ocl;
    ocl.init(opt.aocx_filename, opt.platform, opt.device);

    test_single(ocl, opt.iterations, opt.size);

    ocl.clean();

    return 0;
}