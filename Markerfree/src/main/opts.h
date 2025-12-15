#ifndef OPTS_H__
#define OPTS_H__

#include <iostream>
#include <sstream>
#include <cassert>
extern "C"
{
#include <getopt.h>
}
#include <string>
#include <cstdio>
#include <cstdlib>
#include "util/exception.h"

struct options
{
    char input[255];
    char output[255];
    char initial[255];
    char angle[255];

    // geometry
    float pitch_angle;
    float zshift;
    int thickness;
    int AlignZ;
    int OutBin;
    float offset;
    int GPU;

    int Savemode;
    int nProj; // <--- 新增 - New

    // method
    std::string method;

    // iteration params
    int iteration;
    float gamma;
};

inline void UsageDual()
{
    std::cout << "[-i INPUT FILENAME]\n";
    std::cout << "    MRC file for reconstruction\n";
    std::cout << "[-o OUTPUT FILENAME]\n";
    std::cout << "    MRC filename for result\n";
    std::cout << "[-a TILT ANGLE FILENAME]\n";
    std::cout << "    Tilt Angles\n";
    std::cout << "[-n INITIAL RECONSTRUCTION]\n";
    std::cout << "    MRC file as initial model (optional)\n";
    std::cout << "[-g O,P,Z,T,A,B,G]\n";
    std::cout << "    Geometry: offset,pitch_angle,zshift,thickness,AlignZ,OutBin,GPU\n";
    std::cout << "[-m METHODS (I,R)]\n";
    std::cout << "    BPT\n";
    std::cout << "    SART: SART iteration_number,relax_parameter\n";
    std::cout << "    RP\n";
    std::cout << "[-p NPROJ]\n";
    std::cout << "    Number of neighbor projections\n";
    std::cout << "[-s SAVE_MODE]\n";
    std::cout << "    SaveParameter (0 or 1)\n";
    std::cout << "[-h]\n";
    std::cout << "    Help\n";
}

inline void InitOpts(options *opt)
{
    opt->pitch_angle = 0;
    opt->zshift = 0;
    opt->thickness = 0;
    opt->offset = 0;
    opt->AlignZ = 600;
    opt->OutBin = 1;
    opt->GPU = 0;
    opt->Savemode = 0;
    opt->nProj = 10; // default
    opt->input[0] = '\0';
    opt->output[0] = '\0';
    opt->initial[0] = '\0';
}

inline void PrintOpts(const options &opt)
{
    std::cout << "pitch_angle = " << opt.pitch_angle << "\n";
    std::cout << "zshift = " << opt.zshift << "\n";
    std::cout << "thickness = " << opt.thickness << "\n";
    std::cout << "offset = " << opt.offset << "\n";
    std::cout << "AlignZ = " << opt.AlignZ << "\n";
    std::cout << "OutBin = " << opt.OutBin << "\n";
    std::cout << "GPU = " << opt.GPU << "\n";
    std::cout << "Savemode = " << opt.Savemode << "\n";
    std::cout << "nProj = " << opt.nProj << "\n";
    std::cout << "input = " << opt.input << "\n";
    std::cout << "output = " << opt.output << "\n";
    std::cout << "initial = " << opt.initial << "\n";
    std::cout << "method = " << opt.method << "\n";
    std::cout << "iter = " << opt.iteration << "\n";
    std::cout << "step = " << opt.gamma << "\n";
}

inline int GetOpts(int argc, char **argv, options *opts_)
{

    static struct option longopts[] = {
        {"help", no_argument, NULL, 'h'},
        {"input", required_argument, NULL, 'i'},
        {"output", required_argument, NULL, 'o'},
        {"initial", required_argument, NULL, 'n'},
        {"angle", required_argument, NULL, 'a'},
        {"geometry", required_argument, NULL, 'g'},
        {"mode", required_argument, NULL, 'm'},
        {"savemode", required_argument, NULL, 's'},
        {"nproj", required_argument, NULL, 'p'},
        {NULL, 0, NULL, 0}};

    int ch;
    while ((ch = getopt_long(argc, argv, "hi:o:n:a:g:m:s:p:", longopts, NULL)) != -1)
    {
        switch (ch)
        {
        case '?':
            EX_TRACE("Invalid option '%s'.", argv[optind - 1]);
            return -1;
        case ':':
            EX_TRACE("Missing option argument for '%s'.", argv[optind - 1]);
            return -1;

        case 'h':
            UsageDual();
            return 0;

        case 'i':
        {
            std::istringstream(optarg) >> opts_->input;
            break;
        }
        case 'o':
        {
            std::istringstream(optarg) >> opts_->output;
            break;
        }
        case 'n':
        {
            std::istringstream(optarg) >> opts_->initial;
            break;
        }
        case 'a':
        {
            std::istringstream(optarg) >> opts_->angle;
            break;
        }

        case 'g':
        {
            std::istringstream iss(optarg);
            std::string tmp;
            if (!std::getline(iss, tmp, ','))
                return -1;
            opts_->offset = atof(tmp.c_str());
            if (!std::getline(iss, tmp, ','))
                return -1;
            opts_->pitch_angle = atof(tmp.c_str());
            if (!std::getline(iss, tmp, ','))
                return -1;
            opts_->zshift = atof(tmp.c_str());
            if (!std::getline(iss, tmp, ','))
                return -1;
            opts_->thickness = atoi(tmp.c_str());
            if (!std::getline(iss, tmp, ','))
                return -1;
            opts_->AlignZ = atoi(tmp.c_str());
            if (!std::getline(iss, tmp, ','))
                return -1;
            opts_->OutBin = atoi(tmp.c_str());
            if (!std::getline(iss, tmp))
                return -1;
            opts_->GPU = atoi(tmp.c_str());
            break;
        }

        case 's':
            opts_->Savemode = atoi(optarg);
            if (opts_->Savemode != 0 && opts_->Savemode != 1)
            {
                EX_TRACE("Savemode must be 0 or 1.");
                return -1;
            }
            break;

        case 'p':
        {
            opts_->nProj = atoi(optarg);
            if (opts_->nProj <= 0)
            {
                EX_TRACE("nProj must be > 0.");
                return -1;
            }
            break;
        }

        case 'm':
        {
            std::istringstream iss(optarg);
            std::string tmp;

            if (strcmp(optarg, "BPT") != 0 && strcmp(optarg, "RP") != 0)
            {
                if (!std::getline(iss, opts_->method, ','))
                {
                    EX_TRACE("Invalid method argument.");
                    return -1;
                }
                if (opts_->method == "SIRT" || opts_->method == "SART")
                {
                    if (!std::getline(iss, tmp, ','))
                    {
                        EX_TRACE("Missing iteration number.");
                        return -1;
                    }
                    opts_->iteration = atoi(tmp.c_str());
                    if (!std::getline(iss, tmp))
                    {
                        EX_TRACE("Missing relax parameter.");
                        return -1;
                    }
                    opts_->gamma = atof(tmp.c_str());
                }
            }
            else
            {
                opts_->method = optarg;
            }
            break;
        }

        default:
            assert(false);
        }
    }

    return 1;
}

#endif
