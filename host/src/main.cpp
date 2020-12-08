#include <iostream>
#include <iomanip>
#include <utility>
#include <stdlib.h>
#include <math.h>

#include "opencl.hpp"
#include "common.hpp"
#include "options.hpp"
#include "buffers.hpp"
#include "utils.hpp"

using namespace std;

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

void check_computation(const float * src, const float * dst, int n)
{
    for (int i = 0; i < n; ++i) {
        const float v = src[i] * src[i];
        if (fabsf(dst[i] - v) > FLT_EPSILON) {
            cerr << "ERROR: " << v << " != " << dst[i] << endl;
            exit(-2);
        }
    }
}

void print_results(int iterations, int size,
                   uint64_t t_start,
                   uint64_t t_end,
                   cl_ulong t_reader,
                   cl_ulong t_compute,
                   cl_ulong t_writer,
                   cl_ulong t_read,
                   cl_ulong t_write)
{
    // All timings are in nanoseconds but printed in milliseconds
    cl_ulong t_host     = (t_end - t_start);
    double tavg_reader  = t_reader  / (double)iterations;
    double tavg_compute = t_compute / (double)iterations;
    double tavg_writer  = t_writer  / (double)iterations;
    double tavg_read    = t_read    / (double)iterations;
    double tavg_write   = t_write   / (double)iterations;

    size_t total_bytes  = iterations * size * sizeof(float);
    double bw_reader    = total_bytes / (double)t_reader;
    double bw_compute   = total_bytes / (double)t_compute * 2;
    double bw_writer    = total_bytes / (double)t_writer;
    double bw_read      = total_bytes / (double)t_read;
    double bw_write     = total_bytes / (double)t_write;

    cout << right << fixed  << setprecision(4)
         << "Total time Host (ms): " << setw(10) << t_host * 1.0e-6 << "\n"
         << "┌──────────────────┬────────────┬────────────┬────────────┬────────────┬────────────┐\n"
         << "│                  │   reader   │  compute   │   writer   │    read    │   write    │\n"
         << "├──────────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤\n"
         << "│  Total Time (ms) │ " << setw(10) << t_reader     * 1.0e-6 << " │ "
                                    << setw(10) << t_compute    * 1.0e-6 << " │ "
                                    << setw(10) << t_writer     * 1.0e-6 << " │ "
                                    << setw(10) << t_read       * 1.0e-6 << " │ "
                                    << setw(10) << t_write      * 1.0e-6 << " │\n"
         << "│    Avg Time (ms) │ " << setw(10) << tavg_reader  * 1.0e-6 << " │ "
                                    << setw(10) << tavg_compute * 1.0e-6 << " │ "
                                    << setw(10) << tavg_writer  * 1.0e-6 << " │ "
                                    << setw(10) << tavg_read    * 1.0e-6 << " │ "
                                    << setw(10) << tavg_write   * 1.0e-6 << " │\n"
         << "│ Bandwidth (GB/s) │ " << setw(10) << bw_reader             << " │ "
                                    << setw(10) << bw_compute            << " │ "
                                    << setw(10) << bw_writer             << " │ "
                                    << setw(10) << bw_read               << " │ "
                                    << setw(10) << bw_write              << " │\n"
         << "└──────────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘\n\n";
}

void benchmark(OCL & ocl,
               int iterations,
               int size,
               clKernelType kernel_type,
               clMemoryType mem_type,
               bool check_results = false)
{

    cout << "Benchmark with "
         << (kernel_type == clKernelType::Task ? "clEnqueueTask()" : "clEnqueueNDRangeKernel()")
         << " using "
         << (mem_type == clMemoryType::Buffer ? "clMemBuffer" : "clMemShared")
         << " memory type\n";


     // Queues
    cl_command_queue queues[3];
    queues[0] = ocl.createCommandQueue();
    queues[1] = ocl.createCommandQueue();
    queues[2] = ocl.createCommandQueue();


     // Buffers
    clMemory<float> * src;
    clMemory<float> * dst;

    if (mem_type == clMemoryType::Buffer) {
        src = new clMemBuffer<float>(ocl.context, queues[0], size, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY);
        dst = new clMemBuffer<float>(ocl.context, queues[2], size, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY);
    } else { // clMemoryType::Shared
        src = new clMemShared<float>(ocl.context, queues[0], size, CL_MEM_READ_ONLY);
        dst = new clMemShared<float>(ocl.context, queues[2], size, CL_MEM_WRITE_ONLY);
        
        cl_event event_map[2];
        src->map(CL_MAP_WRITE, &event_map[0]);
        dst->map(CL_MAP_READ, &event_map[1]);

        cout << "src->map(): " << clTimeEventMS(event_map[0]) << " ms\n"
             << "dst->map(): " << clTimeEventMS(event_map[1]) << " ms\n";

        clReleaseEvent(event_map[0]);
        clReleaseEvent(event_map[1]);
    }


    // Kernels
    cl_kernel kernels[3];
    if (kernel_type == clKernelType::Task) {
        kernels[0] = ocl.createKernel(K_READER_SINGLE_NAME);
        kernels[1] = ocl.createKernel(K_COMPUTE_SINGLE_NAME);
        kernels[2] = ocl.createKernel(K_WRITER_SINGLE_NAME);
    } else {
        kernels[0] = ocl.createKernel(K_READER_RANGE_NAME);
        kernels[1] = ocl.createKernel(K_COMPUTE_RANGE_NAME);
        kernels[2] = ocl.createKernel(K_WRITER_RANGE_NAME);
    }

    cl_int argi = 0;
    clCheckError(clSetKernelArg(kernels[0], argi++, sizeof(src->buffer), &src->buffer));
    clCheckError(clSetKernelArg(kernels[0], argi++, sizeof(size), &size));
    argi = 0;
    clCheckError(clSetKernelArg(kernels[1], argi++, sizeof(size), &size));
    argi = 0;
    clCheckError(clSetKernelArg(kernels[2], argi++, sizeof(dst->buffer), &dst->buffer));
    clCheckError(clSetKernelArg(kernels[2], argi++, sizeof(size), &size));


    // Benchmark
    size_t gws[3] = {1, 1, 1};
    size_t lws[3] = {1, 1, 1};
    if (kernel_type == clKernelType::NDRange) {
        gws[0] = size;
        lws[0] = 16;
    }

    // 0-2 kernel times, 3 read time, 4 write time
    cl_ulong timings[5] = {0, 0, 0, 0, 0};
    cl_ulong time_start = current_time_ns();

    for (int i = 0; i < iterations; ++i) {
        cl_event events[5];

        if (mem_type == clMemoryType::Buffer) {
            random_fill(src->ptr, size);
            src->write(&events[4]);
        } else { // clMemoryType::Shared
            // const uint64_t t_write_start = current_time_ns();
            random_fill(src->ptr, size);
            // const uint64_t t_write_end = current_time_ns();
            // timings[4] = t_write_end - t_write_start;
        }

        for (int i = 0; i < 3; ++i) {
            clCheckError(clEnqueueNDRangeKernel(queues[i], kernels[i],
                                                1, NULL, gws, lws,
                                                0, NULL, &events[i]));
        }

       if (mem_type == clMemoryType::Buffer) dst->read(&events[3]);

        for (int i = 0; i < 3; ++i) clFinish(queues[i]);
        for (int i = 0; i < 3; ++i) timings[i] += clTimeEventNS(events[i]);
        for (int i = 0; i < 3; ++i) clReleaseEvent(events[i]);

        if (mem_type == clMemoryType::Buffer) {
            timings[3] = clTimeEventNS(events[3]);
            timings[4] = clTimeEventNS(events[4]);
            clReleaseEvent(events[3]);
            clReleaseEvent(events[4]);
        } else { // clMemoryType::Shared
            // uint64_t t_read_start = current_time_ns();
            // if (check_results) check_computation(src->ptr, dst->ptr, size);
            // uint64_t t_read_end = current_time_ns();
            // timings[3] = t_read_end - t_read_start;
        }
        if (check_results) check_computation(src->ptr, dst->ptr, size);
    }
    for (int i = 0; i < 3; ++i) clFinish(queues[i]);
    cl_ulong time_end = current_time_ns();

    print_results(iterations, size, time_start, time_end,
                  timings[0], timings[1], timings[2],
                  timings[3], timings[4]);


    // Releases
    src->release();
    dst->release();

    delete src;
    delete dst;

    for (int i = 0; i < 3; ++i) if (kernels[i]) clReleaseKernel(kernels[i]);
    for (int i = 0; i < 3; ++i) if (queues[i]) clReleaseCommandQueue(queues[i]);
}

int main(int argc, char * argv[])
{
    Options opt;
    opt.process_args(argc, argv);

    OCL ocl;
    ocl.init(opt.aocx_filename, opt.platform, opt.device);

    double mem_batch = opt.size * sizeof(float) / (double)(1 << 20);
    double mem_total = opt.iterations * mem_batch;
    cout << fixed << setprecision(3)
         << "   Iterations: " << opt.iterations            << "\n"
         << "  Batch Items: " << opt.size                  << " items\n"
         << " Batch Memory: " << mem_batch                 << " MB\n"
         << "  Total Items: " << opt.iterations * opt.size << " items\n"
         << " Total Memory: " << mem_total                 << " MB\n"
         << "\n";


    if (opt.task) {
        if (opt.buffer) benchmark(ocl, opt.iterations, opt.size,
                                  clKernelType::Task, clMemoryType::Buffer,
                                  opt.check_results);
        if (opt.shared) benchmark(ocl, opt.iterations, opt.size,
                                  clKernelType::Task, clMemoryType::Shared,
                                  opt.check_results);
    }

    if (opt.range) {
        if (opt.buffer) benchmark(ocl, opt.iterations, opt.size,
                                  clKernelType::NDRange, clMemoryType::Buffer,
                                  opt.check_results);
        if (opt.shared) benchmark(ocl, opt.iterations, opt.size,
                                  clKernelType::NDRange, clMemoryType::Shared,
                                  opt.check_results);
    }

    ocl.clean();

    return 0;
}