#include "Util.h"
#include "../mrc/mrcstack.h"
#include <stdio.h>
#include <cufft.h>
#include <math.h>
#include <assert.h>

static __device__ float mGBilinear(float XY[2], int width, int height, float* inproj)
{	int iX = (int)XY[0];
    int iY = (int)XY[1];
    int i = iY * width + iX;
	//----------------------
	XY[0] -= iX;
	XY[1] -= iY;
	float f2 = 1.0f - XY[0];
	float f3 = 1.0f - XY[1];
	f2 = inproj[i] * f2 * f3 + inproj[i+1] * XY[0] * f3
		+ inproj[i+width] * f2 * XY[1]
		+ inproj[i+width+1] * XY[0] * XY[1];
	return f2;
}

static __device__ float Grandomfill(float XY[2], int width, int height, float* inproj)
{
    int x = (int)fabsf(XY[0]);  //取绝对值的整数部分 - Take the integer part of the absolute value
	int y = (int)fabsf(XY[1]);
	// if(x >= gridDim.x) x = 2 * gridDim.x - x;
	// if(y >= height) y = 2 * height - y;
    if(x >= width) x = width - 1 - (x % (width-1));
	if(y >= height) y = height - 1 - (y % (height-1));
	//-----------------------------------------------------------
	int iWin = 31, ix = 0, iy = 0;
	int iSize = iWin * iWin;
	unsigned int next = y * width + x;
	for(int i=0; i<iSize; i++)
	{	next = (next * 7) % iSize;
		ix = (next % iSize) - iWin / 2 + x;
		if(ix < 0 || ix >= width) continue;
		//-----------------------------------
		iy = (next / iWin) - iWin / 2 + y;
		if(iy < 0 || iy >= height) continue;
		//---------------------------------------
		return inproj[iy * width + ix];
	}
	return inproj[height / 2 * width + gridDim.x / 2];
}

__global__ void GSum2D(float* proj, int width, int height, int padX, int iExp, float* databuf)
{
    extern __shared__ float shared[];
    float sum = 0.0f;
    for (int y=blockIdx.x; y<height; y+=gridDim.x) 
    {
        float *ptr = proj + y * padX;
        for (int ix=threadIdx.x; ix<width; ix+=blockDim.x) 
		{	float val = ptr[ix];
			if(val < (float)-1e10) continue;
			//------------------------------
			float expval = val;
			for(int i=1; i<iExp; i++)
			{	expval *= val;
			}
            		sum += expval;
          	}
    }

    shared[threadIdx.x] = sum;
	__syncthreads();

    for (int offset=blockDim.x>>1; offset>0; offset>>=1) 
	{	if (threadIdx.x < offset)
		{	shared[threadIdx.x] += shared[threadIdx.x+offset];
		}
		__syncthreads();
	}
    if(threadIdx.x == 0) databuf[blockIdx.x] = shared[0] / (width * height);
}

__global__ void GSum1D(float* databuf)
{
    extern __shared__ float shared[];
    shared[threadIdx.x] = databuf[threadIdx.x];
    __syncthreads();

    for (int offset=blockDim.x>>1; offset>0; offset>>=1) 
    {	if (threadIdx.x < offset)
        {	shared[threadIdx.x] += shared[threadIdx.x+offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) databuf[0] = shared[0];
}

// Code inspired in AreTomo2 source
__global__ void GConv(cufftComplex* gComp1, cufftComplex* gComp2, int iCmpY)
{	
    int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iCmpY) return;
	int i = y * gridDim.x + blockIdx.x;
	//---------------------------------
	float fRe, fIm;
	fRe = gComp1[i].x * gComp2[i].x + gComp1[i].y * gComp2[i].y;   //将gcomp1取共轭了 - gComp1 is conjugated
	fIm = gComp1[i].x * gComp2[i].y - gComp1[i].y * gComp2[i].x;
	//-----------------------------------------------------------
	gComp2[i].x = fRe;
	gComp2[i].y = fIm;
}

__global__ void GWiener(cufftComplex* gComp, float fBFactor, int iCmpY)
{   
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(y >= iCmpY) return;
    int i = y * gridDim.x + blockIdx.x;
    //---------------------------------
	if(y > (iCmpY / 2)) y -= iCmpY;
	float fNx = 2.0f * (gridDim.x - 1);   //原始图像的长度尺寸 - Original image length size
    float fFilt = -2.0f * fBFactor /(fNx * fNx + iCmpY * iCmpY);
    fFilt = expf(fFilt * (blockIdx.x * blockIdx.x + y * y));
	//------------------------------------------------------
	float fAmp = sqrtf(gComp[i].x * gComp[i].x + gComp[i].y * gComp[i].y);
	fFilt /= sqrtf(fAmp + 0.01f);
	gComp[i].x *= fFilt;
	gComp[i].y *= fFilt;
}

__global__ void GCenterOrigin(cufftComplex* gComp, int iCmpY)
{
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iCmpY) return;
	int i = y * gridDim.x + blockIdx.x;
	//---------------------------------
	int iSign = ((blockIdx.x + y) % 2 == 0) ? 1 : -1;
	gComp[i].x *= iSign;
	gComp[i].y *= iSign;
}

__global__ void GNormalize(float* gfImg, int iPadX, int iSizeY, float fMean, float fStd)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iSizeY) return;
	//---------------------
	int i = y * iPadX + blockIdx.x;
	float fInt = gfImg[i];
	if(fInt < (float)-1e10) return;
	//-----------------------------
	gfImg[i] = (fInt - fMean) / fStd;  //每个像素减去均值再除以标准差，经典的归一化操作 
    // Each pixel minus the mean and then divided by the standard deviation, a classic normalization operation
}

__global__ void GStretchRandom(float* inproj, int padX, int height, float* Matrix, float* outproj, bool Randfill)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(y>=height) return;
    int i = y * padX + blockIdx.x;

    float XY[2];
    XY[0] = blockIdx.x - 0.5f * gridDim.x + 0.5f;
    XY[1] = y - 0.5 * height + 0.5f;

    float buf = XY[0] * Matrix[0] + XY[1] * Matrix[1];
    XY[1] = XY[0] * Matrix[1] + XY[1] * Matrix[2] + 0.5f * height - 0.5f;
    XY[0] = buf + 0.5f * gridDim.x - 0.5f;

    if(XY[0]>=0 && XY[0]<gridDim.x - 1 && XY[1]>=0 && XY[1]<height - 1)  //对于变换后在范围内的坐标，直接赋值并退出即可
    // For coordinates that are within the range after transformation, just assign and exit
    {
        outproj[i] = mGBilinear(XY, padX, height, inproj);
        return;
    }
    if(Randfill) outproj[i] = Grandomfill(XY, padX, height, inproj);
    else outproj[i] = (float)(-1e30);
}

__global__ void GRoundEdge(float* proj, int padX, int height, float MaskCentX, float MaskCentY,
                            float MaskSizeX, float MaskSizeY, float Power)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= height) return;
	int i = y * padX + blockIdx.x;
	if(proj[i] < (float)-1e10)
	{	proj[i] = 0.0f;
		return;
	}

    float fx = 2 * fabsf(blockIdx.x - MaskCentX) / MaskSizeX;
    float fy = 2 * fabsf(y - MaskCentY) / MaskSizeY;
    float fr = sqrtf(fx * fx + fy * fy);
    if(fr >= 1.0f)
    {
        proj[i] = 0.0f;
        return;
    }
    fr = 0.5f * (1 - cosf(3.1415926f * fr));
    fr = 1.0f - powf(fr, Power);
    proj[i] = proj[i] * fr;
}

CalcCC::CalcCC()
{
    PadRefProj = 0L;
}

CalcCC::~CalcCC()
{
    if(PadRefProj != 0L) cudaFree(PadRefProj);
    PadRefProj = 0L;
    m_fft.DestroyPlan();
    in_fft.DestroyPlan();
}

void CalcCC::Setup(int* pBinSize, int Factor)
{
	m_Factor = Factor;
	BinSize[0] = pBinSize[0];
	BinSize[1] = pBinSize[1];
	PadSize[0] = (BinSize[0] / 2 + 1) * 2;
    PadSize[1] = BinSize[1];
    CmpSize[0] = PadSize[0] / 2;
    CmpSize[1] = PadSize[1]; 
    //printf("pad和cmpsize:%d,%d,%d,%d\n", PadSize[0], PadSize[1], CmpSize[0],CmpSize[1]);
    int iCmpSize = CmpSize[0] * CmpSize[1];
    size_t tBytes = sizeof(cufftComplex) * iCmpSize * 3;
    cudaMalloc(&PadRefProj, tBytes);
    PadfProj = PadRefProj + iCmpSize;
	stretchProj = PadfProj + iCmpSize;
}

void CalcCC::SetSize(int* pBinSize)
{
    BinSize[0] = pBinSize[0];
	BinSize[1] = pBinSize[1];
	PadSize[0] = (BinSize[0] / 2 + 1) * 2;
    PadSize[1] = BinSize[1];
    CmpSize[0] = PadSize[0] / 2;
    CmpSize[1] = PadSize[1]; 
}

void CalcCC::DoIt(float* RefProj, float* fProj, float fRefTilt, float fTilt, float fTiltAxis)
{
	PadProj(RefProj, (float*)PadRefProj);  //相邻图像的pad和归一化 - Pad and normalize adjacent images
    mNormalize((float*)PadRefProj);
    PadProj(fProj, (float*)PadfProj);   //该图像的pad，拉伸和归一化 - Pad, stretch and normalize this image

    bool bPadded = true; bool randomfill = true;
    double dstretch = cos(D2R * fRefTilt) / cos(D2R * fTilt);
    Stretch((float*)PadfProj, PadSize, bPadded, dstretch, fTiltAxis, (float*)stretchProj, randomfill);
    // Stretch((float*)PadfProj, PadSize, bPadded, dstretch, fTiltAxis, (float*)stretchProj, !randomfill);
    mNormalize((float*)stretchProj);

    float MaskCent[] = {BinSize[0] * 0.5f, BinSize[1] * 0.5f};
    float MaskSize[] = {BinSize[0] * 1.0f, BinSize[1] * 1.0f};
    RoundEdge((float*)PadRefProj, PadSize, bPadded, 4, MaskCent, MaskSize);   //对两张图像运用圆形蒙版 - Apply circular mask to both images
    RoundEdge((float*)stretchProj, PadSize, bPadded, 4, MaskCent, MaskSize);

    bool bNorm = true;
    
    m_fft.CreateForwardPlan(CmpSize);
    if(!m_fft.ForwardFFT((float*)PadRefProj, CmpSize, bNorm))
    {
        printf("执行2D的PadRefProj的fft时失败 - Failed to perform 2D FFT of PadRefProj");
    }
    if(!m_fft.ForwardFFT((float*)stretchProj, CmpSize, bNorm))
    {
        printf("执行2D的stretchProjProj的fft时失败 - Failed to perform 2D FFT of stretchProj");
    }

    int iPixels = (CmpSize[0]-1)*2 * CmpSize[1];
    float* m_pfXcfImg = new float[iPixels];
    getCC(PadRefProj, stretchProj, m_Factor, m_pfXcfImg);
    float Peak = FindPeak(m_pfXcfImg);
    delete[] m_pfXcfImg;
}

void CalcCC::PadProj(float* proj, float* padproj)
{
    size_t tBytes = sizeof(float) * BinSize[0];
    for(int y=0; y<BinSize[1]; y++)
    {
        float *src = proj + y * BinSize[0];
        float *dst = padproj + y * PadSize[0];
        cudaMemcpy(dst, src, tBytes, cudaMemcpyDefault);
    }
}

void CalcCC::mNormalize(float* proj)
{   
    float afMeanStd[] = {0.0f, 1.0f};
    afMeanStd[0] = CalcMoment(proj, BinSize[0], BinSize[1], PadSize[0], 1);
    afMeanStd[1] = CalcMoment(proj, BinSize[0], BinSize[1], PadSize[0], 2);
    afMeanStd[1] = afMeanStd[1] - afMeanStd[0] * afMeanStd[0];
    if(afMeanStd[1] <= 0) afMeanStd[1] = 0.0f;
	else afMeanStd[1] = (float)sqrtf(afMeanStd[1]);
    Norm2D(proj, PadSize[0], PadSize[1], afMeanStd[0], afMeanStd[1]);
}

float CalcCC::CalcMoment(float* proj, int width, int height, int padwidth, int Exponent)
{
    dim3 aBlockDim(512, 1);
	dim3 aGridDim(512, 1);
    int iShmBytes = sizeof(float) * aBlockDim.x;
    float *databuf; float res = 0.0f;
    cudaMalloc(&databuf, sizeof(float) * aGridDim.x);

    GSum2D<<<aGridDim, aBlockDim, iShmBytes>>>(proj, width, height, padwidth, Exponent, databuf);
    GSum1D<<<1, aGridDim, iShmBytes>>>(databuf);
    cudaMemcpy(&res, databuf, sizeof(float), cudaMemcpyDefault);
    cudaFree(databuf);
    return res;
}

void CalcCC::Norm2D(float* proj, int Padwidth, int Padheight, float mean, float std)
{
    dim3 bBlockDim(1, 512);
	dim3 bGridDim((Padwidth / 2 - 1) * 2, 1);
    bGridDim.y = (Padheight + bBlockDim.y - 1) / bBlockDim.y;
    GNormalize<<<bGridDim, bBlockDim>>>(proj, Padwidth, Padheight, mean, std);
}

void CalcCC::Stretch(float* inproj, int* piSize, bool bPadded, float dStretch, float TiltAxis, float *outproj, bool Randfill)
{
    double d2T = 2 * D2R * TiltAxis;
    double sin2T = sin(d2T);
    double cos2T = cos(d2T);
    double P = 0.5 * (dStretch + 1);
    double M = 0.5 * (dStretch - 1);
    float a0 = (float)(P + M * cos2T);
    float a1 = (float)(1 * M * sin2T);
    float a2 = (float)(P - M * cos2T);
    float det = a0 * a2 - a1 * a1;
    float Matrix[3];
    Matrix[0] = a2 / det;
    Matrix[1] = -a1 / det;
    Matrix[2] = a0 / det;
    float* GMatrix;
    cudaMalloc(&GMatrix, sizeof(float) * 3);
    cudaMemcpy(GMatrix, Matrix, sizeof(float) * 3, cudaMemcpyDefault);

    int width = bPadded ? (piSize[0] / 2 - 1) * 2 : piSize[0];  //图像的原始长度 - Original image length
	int height = piSize[1];
    dim3 aBlockDim(1, 512);
	dim3 aGridDim(width, 1);
	aGridDim.y = (height + aBlockDim.y - 1) / aBlockDim.y;
    GStretchRandom<<<aGridDim, aBlockDim>>>
    (inproj, piSize[0], height, GMatrix, outproj,Randfill);
    cudaFree(GMatrix);
}

void CalcCC::RoundEdge(float* Proj, int* piSize, bool bPadded, float fPower, float* MaskCent, float* MaskSize)
{
    int width = bPadded ? (piSize[0] / 2 - 1) * 2 : piSize[0];
    dim3 aBlockDim(1, 512);
	dim3 aGridDim(width, (piSize[1] + aBlockDim.y - 1) / aBlockDim.y);
    GRoundEdge<<<aGridDim, aBlockDim>>>(Proj, piSize[0], piSize[1], MaskCent[0], MaskCent[1],
	  MaskSize[0], MaskSize[1], fPower);
}

void CalcCC::preforCC(cufftComplex* RefProj, cufftComplex* fProj, float Factor)
{
	int aiCmpSize[] = {CmpSize[0], CmpSize[1]};
    dim3 aBlockDim(1, 64);
	dim3 aGridDim(aiCmpSize[0], aiCmpSize[1]/aBlockDim.y + 1);
    GConv<<<aGridDim, aBlockDim>>>(RefProj, fProj, aiCmpSize[1]);
    GWiener<<<aGridDim, aBlockDim>>>(fProj, Factor, aiCmpSize[1]);
    GCenterOrigin<<<aGridDim, aBlockDim>>>(fProj, aiCmpSize[1]); 
}

void CalcCC::getCC(cufftComplex* RefProj, cufftComplex* fProj, float Factor, float* m_pfXcfImg)
{
    preforCC(RefProj, fProj, Factor);

    
    in_fft.CreateInversePlan(CmpSize);
    in_fft.InverseFFT(fProj);
    size_t tBytes = sizeof(float) * (CmpSize[0]-1)*2;
	for(int y=0; y<CmpSize[1]; y++)
	{	float* pfDst = m_pfXcfImg + y * (CmpSize[0]-1)*2;
		float* gfSrc = (float*)(fProj + y * CmpSize[0]);
		cudaMemcpy(pfDst, gfSrc, tBytes, cudaMemcpyDefault);
	}
}

float CalcCC::FindPeak(float* CC)
{
    int searchSize[2] = {0};
    int Peak[2] = {0};
    float fPeak[2] = {0.0f};
    int width = (CmpSize[0]-1)*2;
    int height = CmpSize[1];
    searchSize[0] = width * 8 / 20 * 2;
    searchSize[1] = height * 8 / 20 * 2;
    
    float m_fPeakInt = (float)-1e30;
	//------------------------
	int iStartX = (width - searchSize[0]) / 2;
	int iStartY = (height - searchSize[1]) / 2;
	int iEndX = iStartX + searchSize[0];
	int iEndY = iStartY + searchSize[1];
	for(int y=iStartY; y<iEndY; y++)
	{	int i = y * width;
		for(int x=iStartX; x<iEndX; x++)
		{	if(m_fPeakInt >= CC[i+x]) continue;
			Peak[0] = x;
			Peak[1] = y;
			m_fPeakInt = CC[i+x];
		}
	}
	//-------------------------------------
	if(Peak[0] < 1) Peak[0] = 1;
	else if(Peak[0] > (width - 2)) Peak[0] = width - 2;
	if(Peak[1] < 1) Peak[1] = 1;
	else if(Peak[1] > (height - 2)) Peak[1] = height - 2; 

    int ic = Peak[1] * width + Peak[0];  //当前整数峰值点的位置 - Current integer peak point position
	int xp = ic + 1;
	int xm = ic - 1;
	int yp = ic + width;
	int ym = ic - width;  //这个点的上下左右的点的位置 - The position of the points above, below, left and right of this point
	//--------------------
	double a = (CC[xp] + CC[xm]) * 0.5f - CC[ic];  //ic处二阶导的一半 - Half of the second derivative at ic
	double b = (CC[xp] - CC[xm]) * 0.5f;  //ic处的一阶导 - First derivative at ic
	double c = (CC[yp] + CC[ym]) * 0.5f - CC[ic];
	double d = (CC[yp] - CC[ym]) * 0.5f;
	double dCentX = -b / (2 * a + 1e-30);  //一阶导除以二阶导，用于后续的牛顿法迭代找到一阶导为0的位置 -
    //  The first derivative divided by the second derivative, used for subsequent Newton's method iteration to find the position where the first derivative is 0
	double dCentY = -d / (2 * c + 1e-30);
	//-----------------------------------
	if(fabs(dCentX) > 1) dCentX = 0;  //说明计算错误 - Indicates calculation error
	if(fabs(dCentY) > 1) dCentY = 0;
	fPeak[0] = (float)(Peak[0] + dCentX);
	fPeak[1] = (float)(Peak[1] + dCentY);

    fshiftX = fPeak[0] - width / 2;  //因为图像是经过中心化的，要减去图像的一半得到真实值 
    // - Since the image is centered, half of the image must be subtracted to obtain the true value
	fshiftY = fPeak[1] - height / 2;
    return m_fPeakInt;
}

void CalcCC::getshift(float &shiftX, float &shiftY, float binX, float binY)
{
    shiftX = fshiftX * binX;
    shiftY = fshiftY * binY;
}

// static __global__ void GProjConv(cufftComplex* gComp1, cufftComplex* gComp2, int iCmpY)
// {
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
// 	if(y >= iCmpY) return;
// 	int i = y * gridDim.x + blockIdx.x;
// 	//---------------------------------
// 	float fRe, fIm;
// 	fRe = gComp1[i].x * gComp2[i].x + gComp1[i].y * gComp2[i].y;
// 	fIm = gComp1[i].x * gComp2[i].y - gComp1[i].y * gComp2[i].x;
// 	//----------------------------------------------------------
// 	float fAmp1 = sqrtf(gComp1[i].x * gComp1[i].x
// 		+ gComp1[i].y * gComp1[i].y);
// 	float fAmp2 = sqrtf(gComp2[i].x * gComp2[i].x
// 		+ gComp2[i].y * gComp2[i].y);
// 	float fAmp = sqrtf(fAmp1 * fAmp2) + 0.0001f;
// 	gComp2[i].x = fRe / fAmp;
// 	gComp2[i].y = fIm / fAmp;
// }

// static __global__ void GProjWiener(cufftComplex* gComp, float fBFactor, int iCmpY)
// {       
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
//     if(y >= iCmpY) return;
//     int i = y * gridDim.x + blockIdx.x;
//     //---------------------------------
// 	if(y > (iCmpY / 2)) y -= iCmpY;
// 	float fNx = 2.0f * (gridDim.x - 1);
// 	int ir2 = blockIdx.x * blockIdx.x + y * y;
//     float fFilt = -2.0f * fBFactor /(fNx * fNx + iCmpY * iCmpY);
//     fFilt = expf(fFilt * ir2);
// 	//------------------------
// 	gComp[i].x *= fFilt;
// 	gComp[i].y *= fFilt;
// }

// void CalcCC::ProjCC(cufftComplex* RefProj, cufftComplex* fProj, float Factor)
// {
//     int aiCmpSize[] = {CmpSize[0], CmpSize[1]};
//     dim3 aBlockDim(1, 64);
// 	dim3 aGridDim(aiCmpSize[0], aiCmpSize[1]/aBlockDim.y + 1);
//     GProjConv<<<aGridDim, aBlockDim>>>(RefProj, fProj, aiCmpSize[1]);
//     GProjWiener<<<aGridDim, aBlockDim>>>(fProj, Factor, aiCmpSize[1]);
//     GCenterOrigin<<<aGridDim, aBlockDim>>>(fProj, aiCmpSize[1]); 
// }
