#pragma OPENCL EXTENSION cl_intel_channels : enable
#define DATA_TYPE           float
#define CHANNEL_DEPTH       32
#define WORK_GROUP_SIZE_X   16
#define N_COMPUTE_UNITS     1


// Enqueue Task
channel DATA_TYPE c_reader_compute_s __attribute__((depth(CHANNEL_DEPTH)));
channel DATA_TYPE c_compute_writer_s __attribute__((depth(CHANNEL_DEPTH)));

__attribute__((max_global_work_dim(0)))
__kernel
void reader_single(__global const DATA_TYPE * restrict data, const int n)
{
    for (int i = 0; i < n; ++i) {
        const DATA_TYPE val = data[i];
        write_channel_intel(c_reader_compute_s, val);
    }
}

__attribute__((max_global_work_dim(0)))
__kernel
void compute_single(const int n)
{
    for (int i = 0; i < n; ++i) {
        DATA_TYPE val = read_channel_intel(c_reader_compute_s);
        val = val * val;
        write_channel_intel(c_compute_writer_s, val);
    }
}

__attribute__((max_global_work_dim(0)))
__kernel
void writer_single(__global DATA_TYPE * restrict data, const int n)
{
    for (int i = 0; i < n; ++i) {
        const DATA_TYPE val = read_channel_intel(c_compute_writer_s);
        data[i] = val;
    }
}

// NDRange
channel DATA_TYPE c_reader_compute_r __attribute__((depth(CHANNEL_DEPTH)));
channel DATA_TYPE c_compute_writer_r __attribute__((depth(CHANNEL_DEPTH)));

__attribute__((uses_global_work_offset(0)))
__attribute__((reqd_work_group_size(WORK_GROUP_SIZE_X,1,1)))
__attribute__((num_compute_units(N_COMPUTE_UNITS)))
__kernel
void reader_range(__global const DATA_TYPE * restrict data, const int n)
{
    const int gid = get_global_id(0);

    const DATA_TYPE val = data[gid];
    write_channel_intel(c_reader_compute_r, val);
}

__attribute__((uses_global_work_offset(0)))
__attribute__((reqd_work_group_size(WORK_GROUP_SIZE_X,1,1)))
__attribute__((num_compute_units(N_COMPUTE_UNITS)))
__kernel
void compute_range(const int n)
{
    const int gid = get_global_id(0);

    DATA_TYPE val = read_channel_intel(c_reader_compute_r);
    val = val * val;
    write_channel_intel(c_compute_writer_r, val);
}

__attribute__((uses_global_work_offset(0)))
__attribute__((reqd_work_group_size(WORK_GROUP_SIZE_X,1,1)))
__attribute__((num_compute_units(N_COMPUTE_UNITS)))
__kernel
void writer_range(__global DATA_TYPE * restrict data, const int n)
{
    const int gid = get_global_id(0);

    const DATA_TYPE val = read_channel_intel(c_compute_writer_r);
    data[gid] = val;
}
