#pragma once
#include <stdio.h>
#include <math.h>
#include <vector>
#include <cufft.h>
#include "../mrc/mrcstack.h"

bool mCheckError(cufftResult* pResult, const char* pcFormat);
const char* mGetErrorEnum(cufftResult error);

class cufft2D
{
  public:
    cufft2D();
    ~cufft2D();
    void CreateForwardPlan(int* CmpSize);
    void CreateInversePlan(int* CmpSize);
    bool ForwardFFT(float* proj, int* aiCmpSize, bool bNorm);
    void InverseFFT(cufftComplex* fProj);
    void DestroyPlan();
  private:
    cufftHandle m_plan;
};

class cufft1D
{
    public:
        cufft1D();
        ~cufft1D();
        void DestroyPlan();
        void CreateForwardPlan(int LineSize);   //rotate用的 - for rotate
        void ForwardFFT(float* PadLine);        //rotate用的 - for rotate
        void CreatePlan(int FFTSize, int num, int padsize, bool forward);  //rweight用的 - for rweight
        void Forward(float* Plane);
        void Inverse(cufftComplex* Plane);
    private:
        cufftHandle m_plan;
        cufftType m_cufftType;
        int m_iFFTSize;
        int m_iNumLines;
};
