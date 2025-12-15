#pragma once
#include <stdio.h>
#include <math.h>
#include <vector>
#include <cufft.h>
#include "Util.h"
#include "../mrc/mrcstack.h"
#include "CalcReproj.h"
#include "CorrStack.h"

class ProjAlign
{
    public:
        ProjAlign();
        ~ProjAlign();
        void Setup(MrcStackM &preprojs, AlignParam* pAlignParam, std::vector<float> p_angles, float zoffset, int thickness);
        void Setsize(bool exchange);
        float DoIt();
        void test();
        void SetNProj(int v)
        {
            m_calcreproj.SetNProj(v);
        }

    public:
        int thickness;
        float m_afMaskSize[2];
        const char* outfile;
    private:
        void Clean();
        float DoItBin();
        void RotShift(float* inshift, float angle, float* outshift);
        void GetRotationCenter(float* pfCenter);
        void FitRotCenterZ();
        void FitRotCenterX();
        void RemoveOffsetZ(float fFact);
        void RemoveOffsetX(float fFact);
        void CalcZInducedShift(int iFrame, float* pfShift);
        void CalcXInducedShift(int iFrame, float* pfShift);
        float ClacAlignProj(int i, bool zero = false);
        void MeaAlignProj(int iproj);
        void GetCentral(float* pfImg, float* gfPadImg);
        void PANorm(float* img, float* padimg);
        void projgetCC(cufftComplex* Cmp1, cufftComplex* Cmp2, float Factor, float* m_pfXcfImg); 
        float PAFindPeak(float* CC);
        void setreproj();
        void closereproj();
    private:
        void testreproj();     
        MrcStackM* stack; 
        MrcStackM reproj; 
        AlignParam* param;
        std::vector<float> angles;
        CorrTomoStack* m_pCorrTomoStack;
        CalcReprojFBP m_calcreproj;
        CalcCC gcc;
        cufft2D m_fft;
        cufft2D in_fft;
        float* m_corrprojs;  
        float* mGreproj;  //为pnp方法准备 - Prepared for pnp method
        int BinSize[2];
        int PadBinSize[2];
        int CmpBinSize[2];
        int m_thickness;
        int CentSize[2];
        float m_fZ0;
        float m_fX0;
        bool* SkipProjs;
        float* fReproj;
        float ishiftX;
        float ishiftY;
        float* PadRef;
        float* PadImg;
        int iBin;
        // int iInbin=1;

        float* h_data;
};
