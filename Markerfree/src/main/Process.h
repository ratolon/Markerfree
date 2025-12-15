#pragma once
#include "../method/Util.h"
#include "../mrc/mrcstack.h"
#include "../method/ProjAlign.h"
#include "opts.h"
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>

class Process
{
    public:
        Process();
        ~Process();
        void DoIt(options &opt, SysInfo &info);
    private:
        int ReadStack(options &opt, SysInfo &info);
        std::string extractParentFolder(const char* filename);
        bool ReadAngles(std::vector<float> &angles, const char *name);
        void SetParam();
        void mPreprocess();
        void mCoarseAlign(Geometry &geo);
        void mProjAlign(Geometry &geo, options &opt);
        void ProjAlignOnce(ProjAlign &projalign);
        void ResetShift();
        void CorrectStack(options &opt);
        AlignParam* NewCopyParam();
        void CopyParam(AlignParam* dAlignParam, AlignParam* sAlignParam);
        void CollectParam(options &opt, double time);
    private:
        MrcStackM projs;
        MrcStackM preprojs;
        MrcStackM alignProjs;
        AlignParam* param;
        std::vector<float> p_angles;

};
