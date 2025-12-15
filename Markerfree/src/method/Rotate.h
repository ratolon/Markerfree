#pragma once
#include <stdio.h>
#include <math.h>
#include <vector>
#include <cufft.h>
#include "Util.h"
#include "Cufft1&2.h"
#include "../mrc/mrcstack.h"

class CalcTIltAxis
{
    public:
        CalcTIltAxis();
        ~CalcTIltAxis();
        void Setup(MrcStackM &preprojs, AlignParam* pAlignParam, std::vector<float> p_angles);
        void DoIt(MrcStackM &preprojs, AlignParam* pAlignParam, std::vector<float> p_angles, float angRange, int num);
    private:
        void CLsetup(float angRange, int num);
        void calcComRegion();
        void Radon(int z);
        void FFT1d();
        void CCalcMean(float* PadLine, int LineSize);
        float findTiltAxis();
        float CalcScore(int i);
        void fftSum(cufftComplex* gCmp1, cufftComplex* gCmp2, float fFact1, float fFact2,
	                cufftComplex* gSum, int iCmpSize);
        float GCC1d(cufftComplex* gCmpRef, cufftComplex* gCmpLine);
        int CalcWarps(int iSize, int iWarpSize);
    private:
        MrcStackM* stack;
        AlignParam* param;
        std::vector<float> angles;
        cufft1D m_fft;
        int NumLines;
        float* RotAngles;
        int LineSize;
        int CmpLineSize;
        cufftComplex** CmpPlanes;
        cufftComplex* CmpPlane;
        int* ComRegion;
};
