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

    Options()
    : aocx_filename("./membench.aocx")
    , platform(0)
    , device(0)
    , iterations(32)
    , size(1024)
    {}

    void print_help()
    {
        cout << "-a  --aocx            Specify the path of the .aocx file   \n"
                "-p  --platform        Specify the OpenCL platform index    \n"
                "-d  --device          Specify the OpenCL device index      \n"
                "-i  --iterations      Set the number of iterations         \n"
                "-s  --size            Set the number of items per iteration\n"
                "-h  --help            Show this help message and exit      \n";
        exit(1);
    }

    void process_args(int argc, char * argv[])
    {
        opterr = 0;

        const char * const short_opts = "a:p:d:i:s:h";
        const option long_opts[] = {
                {"aocx",       optional_argument, nullptr, 'a'},
                {"platform",   optional_argument, nullptr, 'p'},
                {"device",     optional_argument, nullptr, 'd'},
                {"iterations", optional_argument, nullptr, 'i'},
                {"size",       optional_argument, nullptr, 's'},
                {"help",       no_argument,       nullptr, 'h'},
                {nullptr,      no_argument,       nullptr,   0}
        };

        int int_opt = -1;

        while (1) {
            const int opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);

            if (opt < 0) break;

            switch (opt) {
                case 'a':
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
                case 's':
                    if ((int_opt = atoi(optarg)) < 0) {
                        cerr << "Please enter a valid number of items per iteration" << endl;
                        exit(1);
                    }
                    size = int_opt;
                    break;
                case 'h':
                case '?':
                default:
                    print_help();
                    break;
            }
        }
    }
};