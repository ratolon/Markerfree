#include "Util.h"
#include "../mrc/mrcstack.h"
#include <stdio.h>
#include <cufft.h>
#include <math.h>

__global__ void FindMin2D(float* proj, int width, int height, float* databuf)
{
    extern __shared__ float shared[];
    float fMin = (float)1e20, fVal = 0.0f;
    for(int y = blockIdx.x; y<height; y+=gridDim.x)
    {   float *ptr = proj + y * width;
        for(int x = threadIdx.x; x<width; x+=blockDim.x)
        {   fVal = ptr[x];
            if(fVal < (float)-1e10) continue;
            else fMin = fminf(fMin, fVal);
        }    
    }
    shared[threadIdx.x] = fMin;
    __syncthreads();
    for(int offset=blockDim.x>>1; offset>0; offset>>=1)
    {   if(threadIdx.x < offset)
        {
            shared[threadIdx.x] = fminf(shared[threadIdx.x],shared[threadIdx.x + offset]);
        }
        __syncthreads();
    }
    if(threadIdx.x == 0) databuf[blockIdx.x] = shared[0];
}

__global__ void FindMax2D(float* proj, int width, int height, float* databuf)
{
    extern __shared__ float shared[];
    float fMax = (float)-1e20, fVal = 0.0f;
    for(int y = blockIdx.x; y<height; y+=gridDim.x)
    {   float *ptr = proj + y * width;
        for(int x = threadIdx.x; x<width; x+=blockDim.x)
        {   fVal = ptr[x];
            if(fVal < (float)-1e10) continue;
            else fMax = fmaxf(fMax, ptr[x]);
        }    
    }
    shared[threadIdx.x] = fMax;
    __syncthreads();
    for(int offset=blockDim.x>>1; offset>0; offset>>=1)
    {   if(threadIdx.x < offset)
        {
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x],shared[threadIdx.x + offset]);
        }
        __syncthreads();
    }
    if(threadIdx.x == 0) databuf[blockIdx.x] = shared[0];
}

__global__ void FindMin1D (float* databuf)
{
    extern __shared__ float shared[];
	shared[threadIdx.x] = databuf[threadIdx.x];
	__syncthreads();
	//--------------
	for (int offset=blockDim.x>>1; offset>0; offset>>=1) 
	{	if (threadIdx.x < offset)
		{	
            shared[threadIdx.x] = fminf(shared[threadIdx.x], shared[threadIdx.x+offset]);
		}
		__syncthreads();
	}
    if (threadIdx.x == 0) databuf[0] = shared[0];
}

__global__ void FindMax1D (float* databuf)
{
    extern __shared__ float shared[];
	shared[threadIdx.x] = databuf[threadIdx.x];
	__syncthreads();
	//--------------
	for (int offset=blockDim.x>>1; offset>0; offset>>=1) 
	{	if (threadIdx.x < offset)
		{	
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x+offset]);
		}
		__syncthreads();
	}
    if (threadIdx.x == 0) databuf[0] = shared[0];
}

__global__ void Subtract(float* proj, int iPixels, float min)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= iPixels) return;
	else if(proj[i] < (float)-1e10) return;
	else proj[i] -= min;
}

PreProcess::PreProcess()
{
}

PreProcess::~PreProcess()
{
}

void PreProcess::SetPositive(void)
{
    printf("Set positivity ...\n");
    printf("  num   residue     min       max\n");
    int iPixels = stack->X() * stack->Y();
    size_t tBytes = sizeof(float) * iPixels;
    float *d_proj, *h_proj;
    cudaMalloc(&d_proj, tBytes);
    h_proj = new float[iPixels];
    float fStackMin = (float)1e30;
    bool m_reverse = angles[0]>angles[1];
    for(int a=0; a<stack->Z(); a++)
    {
        int i;
        if(m_reverse) i = stack->Z() - 1 - a;
        else i = a;
        // i = a;
        rawstack->ReadSlice(a, h_proj);
        
        cudaMemcpy(d_proj, h_proj, tBytes, cudaMemcpyDefault);

        stack->WriteSlice(i, h_proj);
        
        float fmin = FindMinAndMax(d_proj,stack->X(),stack->Y(),MIN);
        float fmax = FindMinAndMax(d_proj,stack->X(),stack->Y(),MAX);
        if(fStackMin > fmin) fStackMin = fmin;
        printf("%4d      %4d    %8.2f  %8.2f\n", i, stack->Z()-i, fmin, fmax); 
    }
    printf("StackMin:%f\n", fStackMin);
    if(fStackMin >= 0)  //如果所有图的最小值大于0，则不用进行额外操作，否则需要进行调整保证所有值为正
    // If the minimum value of all images is greater than 0, no additional operation is required, otherwise adjustment is needed to ensure all values are positive
	{	cudaFree(d_proj);
        delete[] h_proj;
		printf("Positivity set.\n\n");
		return;
	}

    dim3 aBlockDim(512, 1);
	dim3 aGridDim(1, 1);
	aGridDim.x = (iPixels + aBlockDim.x - 1) / aBlockDim.x;
	for(int i=0; i<stack->Z(); i++)
	{	stack->ReadoutputSlice(i, h_proj);
		cudaMemcpy(d_proj, h_proj, tBytes, cudaMemcpyDefault);
		Subtract<<<aGridDim, aBlockDim>>>(d_proj, iPixels, fStackMin);
		cudaMemcpy(h_proj, d_proj, tBytes, cudaMemcpyDefault);
        stack->WriteSlice(i, h_proj);
	}
    cudaFree(d_proj);
    delete[] h_proj;
    printf("Positivity set.\n\n");
}

void PreProcess::MassNormalization()
{
    printf("Linear mass normalization...\n");
    mstart[0] = stack->X() * 1 / 6;
    mstart[1] = stack->Y() * 1 / 6;
    msize[0] = stack->X() * 4 / 6;
    msize[1] = stack->Y() * 4 / 6;
    nz = stack->Z();

    float* pfMean = new float[nz];
    for(int i=0; i<nz; i++)
	{	pfMean[i] = mCalcMean(i);	//得到每张图像非0处的平均像素值 - Get the average pixel value of each image at non-0 locations
	}
    int iZeroTilt = GetFrameIdxFromTilt(stack->Z(), angles, 0.0f);  //选定最接近零倾斜角度的那张图片 - Select the image closest to zero tilt angle
	float RefMean = pfMean[iZeroTilt];
	if(RefMean > 1000.0f) RefMean = 1000.0f; //如果参考平均值超过1000，则定为1000 - If the reference average value exceeds 1000, it is set to 1000
	for(int i=0; i<nz; i++) //通过平均值得到的比例，将所有图像的像素值乘比例 - Multiply the pixel values of all images by the ratio obtained from the average value
	{	float fScale = RefMean / (pfMean[i] + 0.00001f);
		mScale(i, fScale);
	}
	if(pfMean != 0L) delete[] pfMean;
	printf("Linear mass normalization: done.\n\n");
}

float PreProcess::FindMinAndMax(float *proj, int width, int height, MAXIN maxin)
{   
    dim3 blockDim(512,1);
    dim3 gridDim(512,1);
    int iShmBytes = sizeof(float) * blockDim.x;
    float *databuf;
    cudaMalloc(&databuf, sizeof(float) * gridDim.x);
    if(maxin == MIN)
    {
        FindMin2D<<<gridDim, blockDim, iShmBytes>>>(proj, width, height, databuf);
        FindMin1D<<<1, gridDim, iShmBytes>>>(databuf);
    }
    else
    {
        FindMax2D<<<gridDim, blockDim, iShmBytes>>>(proj, width, height, databuf);
        FindMax1D<<<1, gridDim, iShmBytes>>>(databuf);
    }
    float res = 0;
    cudaMemcpy(&res, databuf, sizeof(float), cudaMemcpyDefault);
	return res;
    cudaFree(databuf);
}

// int PreProcess::GetFrameIdxFromTilt(float fTilt)
// {
// 	int iFrameIdx = 0;
// 	float fMin = (float)fabs(fTilt - angles[0]);
// 	for(int i=1; i<stack->Z(); i++)
// 	{	float fDiff = (float)fabs(fTilt - angles[i]);
// 		if(fDiff >= fMin) continue;
// 		fMin = fDiff;
// 		iFrameIdx = i;
// 	}
//     //printf("最接近%f的角度为%f\n", fTilt, angles[iFrameIdx]);
// 	return iFrameIdx;
// }

float PreProcess::mCalcMean(int iFrame)  //计算第iframe张图像的非0处的像素值的平均值 - Calculate the average pixel value of the iframe-th image at non-0 locations
{
	double dMean = 0.0;
	int iCount = 0;
	float* pfFrame = new float[stack->X() * stack->Y()];
    stack->ReadoutputSlice(iFrame, pfFrame);

	int iOffset = mstart[1] * stack->X() + mstart[0];
	for(int y=0; y<msize[1]; y++)
	{	int i = y * stack->X() + iOffset;
		for(int x=0; x<msize[0]; x++)
		{	float fVal = pfFrame[i+x];
			if(fVal <= (float)-1e20) continue;
			dMean += fVal;
			iCount++;
		}
	}
	if(iCount == 0) return 0.0f;
	dMean = dMean / iCount;
    delete[] pfFrame;
	return (float)dMean;
}

void PreProcess::mScale(int iFrame, float fScale)
{   
    int iPixels = stack->X() * stack->Y();
	float* pfFrame = new float[iPixels];
    stack->ReadoutputSlice(iFrame, pfFrame);	
	for(int i=0; i<iPixels; i++)
	{	if(pfFrame[i] <= (float)-1e20) continue;
		pfFrame[i] *= fScale;
	}
    stack->WriteSlice(iFrame, pfFrame);
    delete[] pfFrame;
}
