#pragma once
#include <stdio.h>
#include <math.h>
#include <vector>
#include <cufft.h>
#include "Util.h"
#include "../mrc/mrcstack.h"

struct Coeff
{ // 20个双精度数，前10个是a后10个是b // 20 double precision numbers, the first 10 are a and the last 10 are b
	union
	{
		double p[20];
		struct
		{
			double a[10];
			double b[10];
		};
	};
};

// class Greproj      //in reporj.cu
// {
//     public:
//         Greproj();
//         ~Greproj();
//         void setsize(int projX, int numprojs, int ivolX, int ivolZ);
//         void DoIt(float* fSinogram,float* TiltAngles,int* ProjRange,float projangle);
//         void Clean();
//     public:
//         float* gReproj;
//     private:
//         void BackProj(int* ProjRange, float projangle);
//         void ForwardProj(float projangle);
//     private:
//         int m_aiProjSize[2];
//         int m_aiVolSize[2];
//         float* fvol;
//         float* m_TiltAngles;
//         float* m_fSinogram;
// };

class CalcReprojFBP
{
    public:
        CalcReprojFBP();
        ~CalcReprojFBP();
        void Setup(int* rprojsize, int ivolZ, int num, std::vector<float> angles);
        void DoIt(float* corrproj, bool* SKipProjs, int iProj, float* fReproj);
        void Clean();
        void SetNProj(int v) { nProj = v; }
    private:
        void FindProjRange(std::vector<float> angles, bool* SKipProjs);
        void GetSinogram(int y);
        void Reproj(int y, float projangle);
        void BackProj(float projangle);
        void ForwardProj(float projangle);
    private:
        // Greproj m_greproj;
        std::vector<float> angles;
        int projsize[2];
        int numprojs;
        int nProj = 10;
        float* fSinogram;
        float* TiltAngles;
        int m_aiProjSize[2];
        int m_aiVolSize[2];
        float* gReproj;
        float* fvol;
        float* m_corrprojs;
        float* m_fReproj;
        int m_iProj;
        int ProjRange[2];
        cudaStream_t m_stream = 0;
};
