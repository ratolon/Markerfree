#include "Cufft1&2.h"
#include "../mrc/mrcstack.h"
#include <stdio.h>
#include <cufft.h>
#include <math.h>
#include <assert.h>

__global__ void GMultiplyFactor(cufftComplex* cmp, int cmpY, float Factor)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= cmpY) return;
	int i = y * gridDim.x + blockIdx.x;
	cmp[i].x *= Factor;
	cmp[i].y *= Factor;
}

cufft2D::cufft2D()
{
    m_plan = 0;
}

cufft2D::~cufft2D()
{
    this->DestroyPlan();
}

void cufft2D::CreateForwardPlan(int* CmpSize)
{
	this->DestroyPlan();
    int fftx = (CmpSize[0]-1)*2;
    int ffty = CmpSize[1];
    cufftResult res = cufftPlan2d(&m_plan, ffty, fftx, CUFFT_R2C);
    const char* pcFormat = "2DCreateForwardPlan: %s\n";
	mCheckError(&res, pcFormat);
}

void cufft2D::CreateInversePlan(int* CmpSize)
{
	this->DestroyPlan();
    int fftx = (CmpSize[0]-1)*2;
    int ffty = CmpSize[1];
    cufftResult res = cufftPlan2d(&m_plan, ffty, fftx, CUFFT_C2R);
	const char* pcFormat = "CreateInversePlan: %s\n";
	mCheckError(&res, pcFormat);
}

bool cufft2D::ForwardFFT(float* proj, int* aiCmpSize, bool bNorm)
{
    const char* pcFormat = "Forward: %s\n\n";
	cufftResult res = cufftExecR2C(m_plan, (cufftReal*)proj, (cufftComplex*)proj);
	//----------------------
	if(mCheckError(&res, pcFormat)) return false;
    if(!bNorm) return true;
	
    // int aiCmpSize[] = {CmpSize[0], CmpSize[1]};
	float fFactor = (float)(1.0 / ((aiCmpSize[0]-1)*2) / aiCmpSize[1]);
    dim3 aBlockDim(1, 512);
	int iGridY = (aiCmpSize[1] + aBlockDim.y - 1) / aBlockDim.y;
	dim3 aGridDim(aiCmpSize[0], iGridY);
    GMultiplyFactor<<<aGridDim, aBlockDim>>>((cufftComplex*)proj, aiCmpSize[1], fFactor);
    return true;
}

void cufft2D::InverseFFT(cufftComplex* fProj)
{
    const char* pcFormat2 = "Inverse: %s\n";
	cufftResult res2 = cufftExecC2R(m_plan, fProj, (cufftReal*)fProj);
    mCheckError(&res2, pcFormat2);
}

void cufft2D::DestroyPlan()
{
	if(m_plan == 0) return;
	cufftDestroy(m_plan);
	m_plan = 0;
}

cufft1D::cufft1D()
{
    m_plan = 0;
	m_iFFTSize = 0;
	m_iNumLines = 0;
	m_cufftType = CUFFT_R2C;
}

cufft1D::~cufft1D()
{
    this->DestroyPlan();
}

void cufft1D::CreateForwardPlan(int LineSize)
{
	this->DestroyPlan();
	int ffty = LineSize;
    cufftResult res = cufftPlan1d(&m_plan, ffty, CUFFT_R2C, 1);  //一个图像上的一个投影的的fft变换plan
    const char* pcFormat = "1DCreateForwardPlan: %s\n";
	mCheckError(&res, pcFormat);
}

void cufft1D::ForwardFFT(float* PadLine)
{
	const char* pcFormat = "1DForward: %s\n\n";
	cufftResult res = cufftExecR2C(m_plan, (cufftReal*)PadLine, (cufftComplex*)PadLine);
	//----------------------
	mCheckError(&res, pcFormat);

}

void cufft1D::CreatePlan(int FFTSize, int iNumLines, int padSize, bool forward)
{
	cufftType fftType = forward ? CUFFT_R2C : CUFFT_C2R;
	if(fftType != m_cufftType) this->DestroyPlan();
	else if(m_iFFTSize != FFTSize) this->DestroyPlan();
	else if(m_iNumLines != iNumLines) this->DestroyPlan();
	if(m_plan != 0) return;
	//--------------------------
	m_cufftType = fftType;
	m_iFFTSize = FFTSize;
	m_iNumLines = iNumLines;
	//----------------------
	cufftResult res = cufftPlan1d
	(&m_plan, m_iFFTSize, m_cufftType, m_iNumLines);

	mCheckError(&res, "CreatePlan1D: %s\n\n");
}

// static __global__ void mGMultiply
// (	cufftComplex* gCmpLines, 
// 	int iCmpSize,
// 	float fFactor
// )
// {	int i = blockIdx.x * blockDim.x + threadIdx.x;
//         if(i >= iCmpSize) return;
// 	//-----------------------
// 	int j = blockIdx.y * iCmpSize + i;
// 	gCmpLines[j].x *= fFactor;
// 	gCmpLines[j].y *= fFactor;
// }

void cufft1D::Forward(float* Plane)
{
	const char* pcFormat = "1DForward: %s\n\n";
	cufftResult res = cufftExecR2C
	(m_plan, (cufftReal*)Plane, (cufftComplex*)Plane);
	//----------------------
	mCheckError(&res, pcFormat);

	// int iCmpSize = m_iFFTSize / 2 + 1;
	// dim3 aBlockDim(512, 1);
	// dim3 aGridDim(iCmpSize / aBlockDim.x + 1, m_iNumLines);
	// float fFactor = 1.0f / m_iFFTSize;
	// //--------------------------------
	// mGMultiply<<<aGridDim, aBlockDim>>>
	// ( (cufftComplex*)Plane,
	//   iCmpSize, fFactor
	// );

}

void cufft1D::Inverse(cufftComplex* Plane)
{
	const char* pcFormat2 = "1DInverse: %s\n\n";
	cufftResult res2 = cufftExecC2R(m_plan, Plane, (cufftReal*)Plane);
    mCheckError(&res2, pcFormat2);
}

void cufft1D::DestroyPlan()
{
	if(m_plan == 0) return;
	cufftDestroy(m_plan);
	m_plan = 0;
}

bool mCheckError(cufftResult* pResult, const char* pcFormat)
{
    if(*pResult == CUFFT_SUCCESS) return false;
	//-----------------------------------------
	const char* pcErr = mGetErrorEnum(*pResult);	
	fprintf(stderr, pcFormat, pcErr);
	cudaDeviceReset();
	assert(0);
    return true;
}

const char* mGetErrorEnum(cufftResult error)
{
	switch (error)
    	{	case CUFFT_SUCCESS:
            	return "CUFFT_SUCCESS";
		//---------------------
		case CUFFT_INVALID_PLAN:
            	return "CUFFT_INVALID_PLAN";
		//--------------------------
        	case CUFFT_ALLOC_FAILED:
            	return "CUFFT_ALLOC_FAILED";
		//--------------------------
        	case CUFFT_INVALID_TYPE:
            	return "CUFFT_INVALID_TYPE";
		//--------------------------
        	case CUFFT_INVALID_VALUE:
            	return "CUFFT_INVALID_VALUE";
		//---------------------------
        	case CUFFT_INTERNAL_ERROR:
           	return "CUFFT_INTERNAL_ERROR";
		//----------------------------
        	case CUFFT_EXEC_FAILED:
           	return "CUFFT_EXEC_FAILED";
		//-------------------------
        	case CUFFT_SETUP_FAILED:
            	return "CUFFT_SETUP_FAILED";
		//--------------------------
        	case CUFFT_INVALID_SIZE:
            	return "CUFFT_INVALID_SIZE";
		//--------------------------
        	case CUFFT_UNALIGNED_DATA:
            	return "CUFFT_UNALIGNED_DATA";
    	}
   	return "<unknown>";
}
