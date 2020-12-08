#pragma once

#include <iostream>
#include <iomanip>
#include <string>
#include <getopt.h>

using namespace std;

struct Options
{
    string aocx_filename;
    int platform;
    int device;
    int iterations;
    int size;
    bool task;
    bool range;
    bool autorun;
    bool buffer;
    bool shared;
    bool check_results;

    Options()
    : aocx_filename("./membench.aocx")
    , platform(0)
    , device(0)
    , iterations(32)
    , size(1024)
    , task(false)
    , range(false)
    , buffer(false)
    , shared(false)
    , check_results(false)
    {}

    void print_help()
    {
        cout << "\t-f  --aocx            Specify the path of the .aocx file     \n"
                "\t-p  --platform        Specify the OpenCL platform index      \n"
                "\t-d  --device          Specify the OpenCL device index        \n"
                "\t-i  --iterations      Set the number of iterations           \n"
                "\t-n  --size            Set the number of items per iteration  \n"
                "\t-t  --task            Benchmark clEnqueueTask().             \n"
                "\t-r  --range           Benchmark clEnqueueNDRangeKernel()     \n"
                "\t-a  --autorun         Benchmark Autorun kenrel               \n"
                "\t-b  --buffer          Benchmark clEnqueue[Read/Write]Buffer()\n"
                "\t-s  --shared          Benchmark clEnqueue[Map/Unmap]Buffer() \n"
                "\t-c  --check           Check results of computation           \n"
                "\t-h  --help            Show this help message and exit        \n";
        exit(1);
    }

    void process_args(int argc, char * argv[])
    {
        opterr = 0;

        const char * const short_opts = "f:p:d:i:n:trabsch";
        const option long_opts[] = {
                {"aocx",       optional_argument, nullptr, 'f'},
                {"platform",   optional_argument, nullptr, 'p'},
                {"device",     optional_argument, nullptr, 'd'},
                {"iterations", optional_argument, nullptr, 'i'},
                {"size",       optional_argument, nullptr, 'n'},
                {"task",       optional_argument, nullptr, 't'},
                {"range",      optional_argument, nullptr, 'r'},
                {"buffer",     optional_argument, nullptr, 'b'},
                {"shared",     optional_argument, nullptr, 's'},
                {"check",      optional_argument, nullptr, 'c'},
                {"help",       no_argument,       nullptr, 'h'},
                {nullptr,      no_argument,       nullptr,   0}
        };

        int int_opt = -1;

        while (1) {
            const int opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);

            if (opt < 0) break;

            switch (opt) {
                case 'f':
                    aocx_filename = string(optarg);
                    break;
                case 'p':
                    if ((int_opt = stoi(optarg)) < 0) {
                        cerr << "Please enter a valid platform" << endl;
                        exit(1);
                    }
                    platform = int_opt;
                    break;
                case 'd':
                    if ((int_opt = stoi(optarg)) < 0) {
                        cerr << "Please enter a valid device" << endl;
                        exit(1);
                    }
                    device = int_opt;
                    break;
                case 'i':
                    if ((int_opt = stoi(optarg)) < 0) {
                        cerr << "Please enter a valid number of iterations" << endl;
                        exit(1);
                    }
                    iterations = int_opt;
                    break;
                case 'n':
                    if ((int_opt = atoi(optarg)) < 0) {
                        cerr << "Please enter a valid number of items per iteration" << endl;
                        exit(1);
                    }
                    size = int_opt;
                    break;
                case 't':
                    task = true;
                    break;
                case 'r':
                    range = true;
                    break;
                case 'a':
                    autorun = true;
                    break;
                case 'b':
                    buffer = true;
                    break;
                case 's':
                    shared = true;
                    break;
                case 'c':
                    check_results = true;
                    break;
                case 'h':
                case '?':
                default:
                    print_help();
                    break;
            }
        }

        if (!task and !range and !autorun) {
            cerr << "Please specify at least one of `--task`, `--range` and `--autorun`!\n";
            return;
        }

        if (!buffer and !shared) {
            cerr << "Please specify at least one of `--buffer` and `--shared`!\n";
            return;
        }
    }
};