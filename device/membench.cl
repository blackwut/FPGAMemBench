#pragma OPENCL EXTENSION cl_intel_channels : enable
#define DATA_TYPE           float
#define CHANNEL_DEPTH       32
#define WORK_GROUP_SIZE_X   16


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
__kernel
void reader_range(__global const DATA_TYPE * restrict data, const int n)
{
    const int gid = get_global_id(0);

    const DATA_TYPE val = data[gid];
    write_channel_intel(c_reader_compute_r, val);
}

__attribute__((uses_global_work_offset(0)))
__attribute__((reqd_work_group_size(WORK_GROUP_SIZE_X,1,1)))
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
__kernel
void writer_range(__global DATA_TYPE * restrict data, const int n)
{
    const int gid = get_global_id(0);

    const DATA_TYPE val = read_channel_intel(c_compute_writer_r);
    data[gid] = val;
}

// Autorun
#define DATA_TYPE_VEC float4
#define N_COMPUTE_UNITS 4
channel DATA_TYPE c_reader_compute_a[N_COMPUTE_UNITS] __attribute__((depth(CHANNEL_DEPTH)));
channel DATA_TYPE c_compute_writer_a[N_COMPUTE_UNITS] __attribute__((depth(CHANNEL_DEPTH)));

__attribute__((max_global_work_dim(0)))
__kernel
void reader_autorun(__global const DATA_TYPE_VEC * restrict data, int n)
{
    n = n /4;
    for (int i = 0; i < n; ++i) {
        const DATA_TYPE_VEC val = data[i];
        write_channel_intel(c_reader_compute_a[0], val.s0);
        write_channel_intel(c_reader_compute_a[1], val.s1);
        write_channel_intel(c_reader_compute_a[2], val.s2);
        write_channel_intel(c_reader_compute_a[3], val.s3);
    }
}

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(N_COMPUTE_UNITS)))
__kernel
void compute_autorun()
{
    const int cid = get_compute_id(0);

    while (1) {

        DATA_TYPE val;

        // Read from channel
        switch (cid) {
            case 0: val = read_channel_intel(c_reader_compute_a[0]); break;
            case 1: val = read_channel_intel(c_reader_compute_a[1]); break;
            case 2: val = read_channel_intel(c_reader_compute_a[2]); break;
            case 3: val = read_channel_intel(c_reader_compute_a[3]); break;
        }

        // Computation
        val = val * val;

        // Write to channel
        switch (cid) {
            case 0: write_channel_intel(c_compute_writer_a[0], val); break;
            case 1: write_channel_intel(c_compute_writer_a[1], val); break;
            case 2: write_channel_intel(c_compute_writer_a[2], val); break;
            case 3: write_channel_intel(c_compute_writer_a[3], val); break;
        }
    }
}

__attribute__((max_global_work_dim(0)))
__kernel
void writer_autorun(__global DATA_TYPE_VEC * restrict data, int n)
{
    n = n / 4;
    for (int i = 0; i < n; ++i) {
        DATA_TYPE_VEC val;
        val.s0 = read_channel_intel(c_compute_writer_a[0]);
        val.s1 = read_channel_intel(c_compute_writer_a[1]);
        val.s2 = read_channel_intel(c_compute_writer_a[2]);
        val.s3 = read_channel_intel(c_compute_writer_a[3]);
        data[i] = val;
    }
}
