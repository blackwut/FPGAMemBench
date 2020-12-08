#pragma once
#include "opencl.hpp"

#define AOCL_ALIGNMENT  64

template <typename T>
struct clMemory
{
    cl_context context;
    cl_command_queue queue;
    size_t size;
    cl_mem_flags buffer_flags;

    cl_mem buffer;
    T * ptr;

    clMemory(cl_context context,
             cl_command_queue queue,
             size_t size,
             cl_mem_flags buffer_flags)
    : context(context)
    , queue(queue)
    , size(size)
    , buffer_flags(buffer_flags)
    {}

    virtual void map(cl_map_flags flags, cl_event * event = NULL, bool blocking = true) = 0;
    virtual void read(cl_event * event = NULL, bool blocking = true) = 0;
    virtual void write(cl_event * event = NULL, bool blocking = true) = 0;
    virtual void release() = 0;

    virtual ~clMemory() {};
};

template <typename T>
struct clMemShared: clMemory<T>
{
    using super = clMemory<T>;
    using super::context;
    using super::queue;
    using super::size;
    using super::buffer_flags;
    using super::buffer;
    using super::ptr;

    clMemShared(cl_context context,
                cl_command_queue queue,
                size_t size,
                cl_mem_flags buffer_flags)
    : super(context, queue, size, buffer_flags)
    {}

    void map(cl_map_flags flags,
             cl_event * event = NULL,
             bool blocking = true) override
    {
        cl_int status;
        buffer = clCreateBuffer(context,
                                CL_MEM_ALLOC_HOST_PTR | buffer_flags,
                                size * sizeof(T),
                                NULL, &status);
        clCheckErrorMsg(status, "Failed to create clBufferShared");
        ptr = (T *)clEnqueueMapBuffer(queue, buffer,
                                      blocking, flags,
                                      0, size * sizeof(T),
                                      0, NULL,
                                      event, &status);
        clCheckErrorMsg(status, "Failed to map clBufferShared");
    }

    void read(cl_event * event = NULL, bool blocking = true) override
    {}

    void write(cl_event * event = NULL, bool blocking = true) override
    {}

    void release() override
    {
        if (ptr && buffer) clEnqueueUnmapMemObject(queue, buffer, ptr, 0, NULL, NULL);
        if (buffer) clReleaseMemObject(buffer);
    }
};

template <typename T>
struct clMemBuffer : clMemory<T>
{
    using super = clMemory<T>;
    using super::context;
    using super::queue;
    using super::size;
    using super::buffer_flags;
    using super::buffer;
    using super::ptr;

    clMemBuffer(cl_context context,
                cl_command_queue queue,
                size_t size,
                cl_mem_flags buffer_flags)
    : clMemory<T>(context, queue, size, buffer_flags)
    {
        cl_int status;
        buffer = clCreateBuffer(context, buffer_flags, size * sizeof(T), NULL, &status);
        clCheckErrorMsg(status, "Failed to create clBuffer");

        status = posix_memalign((void**)&ptr, AOCL_ALIGNMENT, size * sizeof(T));
        if (status != 0) clCheckErrorMsg(-255, "Failed to create host buffer");
    }

    void map(cl_map_flags flags, cl_event * event = NULL, bool blocking = true) override
    {}

    void read(cl_event * event = NULL, bool blocking = true) override
    {
        clCheckError(clEnqueueReadBuffer(queue, buffer, blocking,
                                         0, size * sizeof(T), ptr,
                                         0, NULL, event));
    }

    void write(cl_event * event = NULL, bool blocking = true) override
    {
        clCheckError(clEnqueueWriteBuffer(queue, buffer, blocking,
                                          0, size * sizeof(T), ptr,
                                          0, NULL, event));
    }

    void release() override
    {
        if (buffer) clReleaseMemObject(buffer);
        if (ptr) free(ptr);
    }
};
