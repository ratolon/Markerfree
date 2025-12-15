#include "FindOffset.h"
#include "../mrc/mrcstack.h"
#include <stdio.h>
#include <cufft.h>
#include <math.h>

__global__ void GConv2D(float* proj1, float* proj2, int width, int height, float* buf)
{
    extern __shared__ float shared[];
    float afTemp[6] = {0.0f};
	for(int y=blockIdx.x; y<height; y+=gridDim.x)
	{	float* gfPtr1 = proj1 + y * width;
		float* gfPtr2 = proj2 + y * width;
		for(int x=threadIdx.x; x<width; x+=blockDim.x)
		{	float v1 = gfPtr1[x];
			if(v1 < (float)-1e10) continue;
			float v2 = gfPtr2[x];
			if(v2 < (float)-1e10) continue;
			//----------------------------
			afTemp[0] += v1;
			afTemp[1] += v2;
			afTemp[2] += (v1 * v1);
			afTemp[3] += (v2 * v2);
			afTemp[4] += (v1 * v2);
			afTemp[5] += 1.0f;
		}
	}
    for(int i=0; i<6; i++)
	{	shared[threadIdx.x + i * blockDim.x] = afTemp[i];  //blockDim.x，x线程的数量 - number of x threads
	}
	__syncthreads();

    for(int offset=blockDim.x>>1; offset>0; offset>>=1)
	{	if(threadIdx.x < offset)
		{	for(int i=0; i<6; i++)
			{	int j = threadIdx.x + i * blockDim.x;
				shared[j] += shared[j + offset];
			}
		}
		__syncthreads();
	}
    if(threadIdx.x == 0)
	{	for(int i=0; i<6; i++)
		{buf[6 * blockIdx.x + i] = shared[i * blockDim.x];   //将每个block的和放入gfres - Put the sum of each block into gfres
		}
	}
}

__global__ void GConvSum1D(float* buf)
{
    extern __shared__ float shared[];
	for(int i=0; i<6; i++)
	{	shared[i * blockDim.x+threadIdx.x] = buf[6 * threadIdx.x + i];
	}
	__syncthreads(); 

    for(int offset=blockDim.x>>1; offset>0; offset>>=1)
	{	if(threadIdx.x < offset)
		{	for(int i=0; i<6; i++)
			{	int j = i * blockDim.x + threadIdx.x;
				shared[j] += shared[j + offset];
			}
			__syncthreads();
		}
	}
	//------------------------------
	if(threadIdx.x == 0)
	{	for(int i=0; i<6; i++)
		{	buf[i] = shared[i * blockDim.x];
		}
	}
}

FindOffset::FindOffset()
{
    corrstack = 0L;
}

FindOffset::~FindOffset()
{
    if(corrstack != 0L) delete corrstack;
	corrstack = 0L;
}

void FindOffset::DoIt(MrcStackM &preprojs, AlignParam* pAlignParam, std::vector<float> p_angles)
{
    stack = &preprojs;
    param = pAlignParam;
    angles = p_angles;
    FOsetup();
	
    printf("Determine tilt angle offset.\n");
    param->angleOffset += SearchOffset(31, 1.0f, 0.0f);
//     param->angleOffset = 0;
    printf("旋转轴偏移角为: %f\n", param->angleOffset);
}

void FindOffset::FOsetup()
{
    int iBinX = (int)(stack->X() / 1024.0f + 0.1f);
	int iBinY = (int)(stack->Y() / 1024.0f + 0.1f);
	int iBin = (iBinX > iBinY) ? iBinX : iBinY;
    iBin = 4;
	if(iBin < 1) iBin = 1;
    float fTiltAxis = param->rotate[stack->Z() / 2];

    corrstack = new CorrTomoStack;
    bool randomfill = true; bool onlyshift = true; 
    bool rweight = true; bool write = true;  
    corrstack->Set0(stack, param);
    corrstack->Set1(fTiltAxis, iBin);
    corrstack->Set2(!randomfill, !onlyshift, !rweight);
    corrstack->Set3(0L, !write);
    m_corrprojs = corrstack->GetCorrectedProjs();
    corrstack->GetBinning(BinSize);
    corrstack->DoIt();

    // corrstack->stack = stack;
    // corrstack->param = param;
    // bool onlyshift = true; bool randomfill = true; bool write = true;
    // corrstack->SetSize(fTiltAxis, iBin);
    // corrstack->Setup(!onlyshift, !randomfill, !write);
    // corrstack->CorrectProjs();
    // m_corrprojs = corrstack->corrprojs;
    // BinSize[0] = corrstack->BinSize[0];
    // BinSize[1] = corrstack->BinSize[1];
}

float FindOffset::SearchOffset(int num, float step, float initoffset)
{
    float Maxcc = 0.0f;
    float BestOffset = 0.0f;
    for(int i=0; i<num; i++)
    {
        float fOffset = initoffset + (i - num / 2) * step;
        float cc = CalcAverangedCC(fOffset);
        if(cc > Maxcc)
        {
            Maxcc = cc;
            BestOffset = fOffset;
        }
        printf("...... %8.2f  %.4e\n", fOffset, cc);
    }
    printf("Tilt offset: %8.2f,  CC: %.4f\n\n", BestOffset, Maxcc);
	return BestOffset;
}

float FindOffset::CalcAverangedCC(float offset)
{   
    for(int z=0; z<stack->Z(); z++)
    {
        angles[z] += offset;
    }
    
    int count = 0;
    float ccsum = 0.0f;
    int iZeroTilt = GetFrameIdxFromTilt(stack->Z(), angles, 0.0f); 
    for(int i=0; i<stack->Z(); i++)
    {
        // int iRefProj = FindRefIndex(stack->Z(), angles, i);
        // if(iRefProj == i) continue;
        if(i == iZeroTilt) continue;
        int iRefProj = (i < iZeroTilt) ? i+1 : i-1; 
        float cc = Correlate(iRefProj, i);
        ccsum += cc;
        count++;
    }
    float Meancc = ccsum / count;
    for(int z=0; z<stack->Z(); z++)
    {
        angles[z] -= offset;
    }
    return Meancc;
}

float FindOffset::Correlate(int iRefProj, int iProj)
{
    int iPixels = BinSize[0] * BinSize[1];
    // float* Refproj = bincorrect + iRefProj * iPixels;
    // float* Proj = bincorrect + iProj * iPixels;
    float* Refproj = m_corrprojs + iRefProj * iPixels;
    float* Proj = m_corrprojs + iProj * iPixels;
    float RefTilt = angles[iRefProj];
    float Tilt = angles[iProj];
    //float TiltAxis = param->rotate[iProj];

    size_t tBytes = sizeof(float) * BinSize[0] * BinSize[1];
    float *d_refproj, *d_buf, *d_proj;
    cudaMalloc(&d_refproj, tBytes);
    cudaMalloc(&d_buf, tBytes);
    cudaMalloc(&d_proj, tBytes);
    cudaMemcpy(d_buf, Proj, tBytes, cudaMemcpyDefault);
	cudaMemcpy(d_refproj, Refproj, tBytes, cudaMemcpyDefault);

    CalcCC stetch;
    double dStretch = cos(D2R * RefTilt) / cos(D2R * Tilt);
    bool bPadded = true; bool Randfill = true;
    // stetch.Setup(BinSize, 0);
    stetch.Stretch(d_buf, BinSize, !bPadded, (float)dStretch, 0.0f, d_proj, !Randfill);
    //Stretch(d_buf, BinSize, !bPadded, (float)dStretch, TiltAxis, d_proj, Randfill);

    // stetch.mNormalize(d_refproj);
    // stetch.mNormalize(d_proj);

    // float MaskCent[] = {BinSize[0] * 0.5f, BinSize[1] * 0.5f};
    // float MaskSize[] = {BinSize[0] * 1.0f, BinSize[1] * 1.0f};
    // stetch.RoundEdge(d_refproj, BinSize, !bPadded, 4, MaskCent, MaskSize);   //对两张图像运用圆形蒙版 - Apply circular mask to both images
    // stetch.RoundEdge(d_proj, BinSize, !bPadded, 4, MaskCent, MaskSize);

    dim3 aBlockDim(128, 1);
	dim3 aGridDim(128, 1);
    int iShmBytes = sizeof(float) * aBlockDim.x * 6;
    float *databuf;
    cudaMalloc(&databuf, sizeof(float) * aGridDim.x * 6);
    GConv2D<<<aGridDim, aBlockDim, iShmBytes>>>
	(d_refproj, d_proj, BinSize[0], BinSize[1], databuf);
    GConvSum1D<<<1, aGridDim, iShmBytes>>>(databuf);
    float afVals[6] = {0.0f};
	cudaMemcpy(afVals, databuf, sizeof(afVals), cudaMemcpyDefault);

    for(int i=0; i<5; i++) afVals[i] /= afVals[5];
	double dStd1 = afVals[2] - afVals[0];  //img1的方差 - Variance of img1
	double dStd2 = afVals[3] - afVals[1];  //img2的方差 - Variance of img2
	double dCC = afVals[4] - afVals[0] * afVals[1];  //计算归一化互相关 - Calculate normalized cross-correlation
	if(dStd1 < 0) dStd1 = 0;
	if(dStd2 < 0) dStd2 = 0;
	dStd1 = sqrt(dStd1);  //求出标准差 - Get the standard deviation
	dStd2 = sqrt(dStd2);
	dCC = dCC / (dStd1 * dStd2 + 1e-30);  //计算归一化互相关 - Calculate normalized cross-correlation
	if(dCC < -1) dCC = -1;
	else if(dCC > 1) dCC = 1;	

    cudaFree(d_refproj);
    cudaFree(d_proj);
    cudaFree(d_buf);
    cudaFree(databuf);
    return (float)dCC;
}
