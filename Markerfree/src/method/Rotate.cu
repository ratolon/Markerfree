#include "Rotate.h"
#include "../mrc/mrcstack.h"
#include <stdio.h>
#include <cufft.h>
#include <math.h>

extern __shared__ char s_acArray[];

__global__ void GCalcCommonRegion(int width, int height, int LineSize, float* gRotAngles, 
                                    float shiftX, float shiftY, int* ComRegion)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;   //blockIdx.x表示第几个尝试的角度，y表示公共线上的第几个点
	// blockIdx.x indicates the number of attempted angles, y indicates the number of points on the common line
	if(y >= LineSize) return;
	int i = blockIdx.x * LineSize + y;

    float fcos = gRotAngles[blockIdx.x] * 3.1415926f / 180.0f;
    float fsin = sinf(fcos);
    fcos = cosf(fcos);

    int half = LineSize / 2;
    float fOffsetX = width * 0.5f + shiftX;
	float fOffsetY = height * 0.5f + shiftY;
	float fY = y - LineSize * 0.5f;  //当前的y相对于中心的偏移 - The current y offset relative to the center
    for(int x=1; x<half; x++)   //假设旋转图像了，公共线和x轴垂直了，然后以width/2，LineSize/2为中心 - 
	// Assuming the image is rotated, the common line is perpendicular to the x-axis, and then centered at width/2, LineSize/2
	{	float fOldX = -x * fcos - fY * fsin + fOffsetX;
		float fOldY = -x * fsin + fY * fcos + fOffsetY;
		if(fOldX < 0 || fOldX >= width) break;
		if(fOldY < 0 || fOldY >= height) break;
		ComRegion[2 * i] = -x;
	}
	for(int x=1; x<half; x++)
	{	float fOldX = x * fcos - fY * fsin + fOffsetX;
		float fOldY = x * fsin + fY * fcos + fOffsetY;
		if(fOldX < 0 || fOldX >= width) break;
		if(fOldY < 0 || fOldY >= height) break;
		ComRegion[2 * i + 1] = x;
	}
}

__global__ void GRadon(float* proj, int width, int height, float* gRotAngles, float fcos, 
                        float shiftX, float shiftY, int* ComRegion, int LineSize, float* PadLines)
{
    int x, y, i;
	y = blockIdx.y * blockDim.y + threadIdx.y;  //表示一条line上的第几个值，也就是旋转后的y轴坐标 -
	// indicates the number of values on a line, which is the y-axis coordinate after rotation
	if(y >= LineSize) return;
	//-------------------------
	float fCosRot = gRotAngles[blockIdx.x] * 3.1415926f / 180.0f;
	float fSinRot = sinf(fCosRot);
	fCosRot = cosf(fCosRot);
	//----------------------
	float fY = y - LineSize * 0.5f;
	float fSum = 0.0f;
	int iCount = 0;

    i = blockIdx.x * LineSize + y;         //blockIdx.x表示第几条线 - blockIdx.x indicates which line
    float fOffsetX = width * 0.5f + shiftX;
	float fOffsetY = height * 0.5f + shiftY;
    int iLeftWidth = (int)(ComRegion[2 * i] * fcos + 0.5f);  //根据图像投影角度将共同区域进行裁剪 -
	// According to the image projection angle, the common area is cropped
    for(x=iLeftWidth; x<0; x++)
	{	float fOldX = x * fCosRot - fY * fSinRot + fOffsetX;
		float fOldY = x * fSinRot + fY * fCosRot + fOffsetY;
		if(fOldX < 0 || fOldX >= width) continue;
		if(fOldY < 0 || fOldY >= height) continue;
		//----------------------------------------------
		fOldX = proj[((int)fOldY) * width + (int)fOldX];  //此处没有使用插值 - No interpolation is used here
		// fOldX = mGBilinear(fOldX, fOldY, width, height, proj);
		if(fOldX < 0) continue;
		//---------------------
		fSum += fOldX;
		iCount++;
	}
	int iRightWidth = (int)(ComRegion[2*i+1] * fcos + 0.5f);
	for(x=0; x<iRightWidth; x++)
	{	float fOldX = x * fCosRot - fY * fSinRot + fOffsetX;
		float fOldY = x * fSinRot + fY * fCosRot + fOffsetY;
		if(fOldX < 0 || fOldX >= width) continue;
		if(fOldY < 0 || fOldY >= height) continue;
		//----------------------------------------------
		fOldX = proj[((int)fOldY) * width + (int)fOldX];  //此处没有使用插值 - No interpolation is used here
		// fOldX = mGBilinear(fOldX, fOldY, width, height, proj);
		if(fOldX < 0) continue;
		//---------------------
		fSum += fOldX;
		iCount++;
	}
	//------------
	y = blockIdx.x * (LineSize / 2 + 1) * 2 + y;  //用于傅里叶变换的数组，正好可以存储两倍大小的浮点数 -
	// Used for Fourier transform arrays, which can store twice the size of floating point numbers
	if(iCount == 0) PadLines[y] = (float)-1e30;
	else PadLines[y] = fSum / iCount;  //要除以像素数量 - Must be divided by the number of pixels
}

__global__ void GCalcMean(float* gfPadLine, int iSize)    //重新写的 Mean calculation - rewritten kernel
{
	__shared__ float sfSum[512];
	__shared__ int siCount[512];
	sfSum[threadIdx.x] = 0.0f;
	siCount[threadIdx.x] = 0;
	__syncthreads();
	//--------------
	float sum = 0.0f;
	int count = 0;
	for(int x=threadIdx.x; x<iSize; x+=blockDim.x)
	{
		float fVal = gfPadLine[x];
		if(fVal > (float)-1e10)
		{	sum += fVal;
			count += 1;
		}
	}
	
	sfSum[threadIdx.x] = sum;
	//printf("sum: %f", sfSum[threadIdx.x]);
	siCount[threadIdx.x] = count;
	__syncthreads();
	//----------------------
	for (int offset=blockDim.x>>1; offset>0; offset>>=1) 
    {	if (threadIdx.x < offset)
        {	sfSum[threadIdx.x] += sfSum[threadIdx.x+offset];
			siCount[threadIdx.x] += siCount[threadIdx.x+offset];
        }
        __syncthreads();
    }
	//------------
	// if(threadIdx.x == 0)
	// {	if(siCount[0] == 0) databuf[blockIdx.x] = 0;
	// 	else {
	// 		databuf[blockIdx.x] = sfSum[0];
	// 		countbuf[blockIdx.x] = siCount[0];
	// 	}
	// }
	if(threadIdx.x == 0)
	{	//printf("sum: %f, count: %d", sfSum[0], siCount[0]);
		if(siCount[0] == 0) gfPadLine[iSize+1] = 0;
		else gfPadLine[iSize+1] = sfSum[0] / siCount[0];
	}
}

// Inspired in AreTomo2
__global__ void GRemoveMean(float* gfPadLine, int iSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= iSize) return;
	//if(gfPadLine[i] < (float)-1e10) gfPadLine[i] = 0;
	if(gfPadLine[i] < 0) gfPadLine[i] = 0.0f;
	else gfPadLine[i] -= gfPadLine[iSize+1];  //每个数都减去均值 - Each number minus the mean
	//--------------------------------------
	float fHalf = 0.5f * iSize;            //滤波，提取相对中间的部分 - Filtering, extracting the part relative to the middle
	float fR = fabsf(i - fHalf) / fHalf;
	fR = 0.5f * (1 - cosf(3.14159 * fR));
	fR = 1.0f - powf(fR, 100.0f);
	gfPadLine[i] *= (fR * fR);
}

__global__ void GSumCmp(cufftComplex* gCmp1, cufftComplex* gCmp2, float fFact1, float fFact2,
                        cufftComplex* gSum, int iCmpSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= iCmpSize) return;
	//-------------------------------------
	if(gCmp1 != 0L && gCmp2 != 0L)
	{	gSum[i].x = gCmp1[i].x * fFact1 + gCmp2[i].x * fFact2;
		gSum[i].y = gCmp1[i].y * fFact1 + gCmp2[i].y * fFact2;
	}
	else if(gCmp1 != 0L)
	{	gSum[i].x = gCmp1[i].x * fFact1;
		gSum[i].y = gCmp1[i].y * fFact1;
	}
	else if(gCmp2 != 0L)
	{	gSum[i].x = gCmp2[i].x * fFact2;
		gSum[i].y = gCmp2[i].y * fFact2;
	}
}

__global__ void GConv(cufftComplex* gComp1, cufftComplex* gComp2, int iCmpSize, 
                        float fBFactor, float* gfCC, float* gfStd)
{   
    float* sfCC = (float*)&s_acArray[0];
	float* sfStd = &sfCC[blockDim.x];
	sfCC[threadIdx.x] = 0.0f;
	sfStd[threadIdx.x] = 0.0f;
	__syncthreads();
	//------------------------
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < iCmpSize && i != 0)
	{	float fFilt = i / (2.0f * (iCmpSize - 1));
		fFilt = expf(-2.0f * fBFactor * fFilt * fFilt);  //高斯滤波处理 - Gaussian filtering
		//---------------------------------------------
		float fA1, fA2;
		fA1 = gComp1[i].x * gComp1[i].x + gComp1[i].y * gComp1[i].y;
		fA2 = gComp2[i].x * gComp2[i].x + gComp2[i].y * gComp2[i].y;
		fA1 = sqrtf(fA1);
		fA2 = sqrtf(fA2);
		//---------------
		sfCC[threadIdx.x] = (gComp2[i].x * gComp1[i].x 
			+ gComp2[i].y * gComp1[i].y) * fFilt 
			/ (fA1 * fA2 + (float)1e-20);
		sfStd[threadIdx.x] = fFilt;
	}
	__syncthreads();
	//--------------
	i = blockDim.x >> 1;
	while(i > 0)
	{	if(threadIdx.x < i)
		{	sfCC[threadIdx.x] += sfCC[i + threadIdx.x];
			sfStd[threadIdx.x] += sfStd[i + threadIdx.x];
		}
		__syncthreads();
		i = i >> 1;
	}
	//-------------
	if(threadIdx.x != 0) return;
	gfCC[blockIdx.x] = sfCC[0];
	gfStd[blockIdx.x] = sfStd[0];
}

__global__ void GSum(float* gfCC, float* gfStd, int iSize)
{
    float* sfCCSum = (float*)&s_acArray[0];
	float* sfStdSum = (float*)&sfCCSum[blockDim.x];
    sfCCSum[threadIdx.x] = 0.0f;
	sfStdSum[threadIdx.x] = 0.0f;
	__syncthreads();
	//---------------------------
	float cc = 0.0f;
	float std = 0;
	for(int x=threadIdx.x; x<iSize; x+=blockDim.x)
	{
		cc += gfCC[x];
		std += gfStd[x];
	}
	sfCCSum[threadIdx.x] = cc;
	sfStdSum[threadIdx.x] = std;
	__syncthreads();
	//--------------
	for (int offset=blockDim.x>>1; offset>0; offset>>=1) 
    {	if (threadIdx.x < offset)
        {	sfCCSum[threadIdx.x] += sfCCSum[threadIdx.x + offset];
			sfStdSum[threadIdx.x] += sfStdSum[threadIdx.x + offset];
        }
        __syncthreads();
    }
	//-----------------
	if(threadIdx.x != 0) return;
	gfCC[0] = sfCCSum[0];
	gfStd[0] = sfStdSum[0];
}

__global__ void GSum1D(float* gfCC, float* gfStd)
{
    extern __shared__ float sfCCSum[];
	extern __shared__ float sfStdSum[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;

	sfCCSum[threadIdx.x] = gfCC[i];
	sfStdSum[threadIdx.x] = gfStd[i];
	
    __syncthreads();

    for (int offset=blockDim.x>>1; offset>0; offset>>=1) 
    {	if (threadIdx.x < offset)
        {	sfCCSum[threadIdx.x] += sfCCSum[threadIdx.x+offset];
			sfStdSum[threadIdx.x] += sfStdSum[threadIdx.x+offset];
        }
        __syncthreads();
    }
    if(threadIdx.x != 0) return;
	gfCC[0] = sfCCSum[0];
	gfStd[0] = sfStdSum[0];
}

CalcTIltAxis::CalcTIltAxis()
{
}

CalcTIltAxis::~CalcTIltAxis()
{
    m_fft.DestroyPlan();
}

void CalcTIltAxis::Setup(MrcStackM &preprojs, AlignParam* pAlignParam, std::vector<float> p_angles)
{
    stack = &preprojs;
    param = pAlignParam;
    angles = p_angles;  //最后看看这里还需要设置什么，要是设置的不多的话直接塞doit中去 - Finally, see what else needs to be set here, if not much, just put it in doit
}

void CalcTIltAxis::DoIt(MrcStackM &preprojs, AlignParam* pAlignParam, std::vector<float> p_angles, float angRange, int num)
{
	stack = &preprojs;
    param = pAlignParam;
    angles = p_angles;
    CLsetup(angRange, num);

    int iBytes = sizeof(int) * NumLines * LineSize * 2;  //ComRegion 是一个 2D 数组,只记录一个图像的数据，在fft之后存储在CmpPlanes中 -
	// ComRegion is a 2D array that only records the data of one image, after fft it is stored in CmpPlanes
    cudaMalloc(&ComRegion, iBytes);
    size_t tBytes = sizeof(cufftComplex) * NumLines * CmpLineSize;  //存储一张图像的所有投影fft数据 - storing all projection fft data of one image
    cudaMalloc(&CmpPlane, tBytes);

    m_fft.CreateForwardPlan(LineSize);

    calcComRegion();

    for(int z=0; z<stack->Z(); z++)
    {
        Radon(z);
        FFT1d();
        cudaMemcpy(CmpPlanes[z], CmpPlane, tBytes, cudaMemcpyDefault);
    }
    float fTiltAxis = findTiltAxis();
	if(fTiltAxis < -45 && param->rotate[0] == 0) 
	{	fTiltAxis += 180.0f;
	}
	if(fTiltAxis > 135)
	{
		fTiltAxis -= 180.0f;
	}
    
	printf("Initial estimate of tilt axes:\n");
	for(int i=0; i<stack->Z(); i++)
	{	param->rotate[i] = fTiltAxis;
	}
	printf("New tilt axis: %.2f\n\n", fTiltAxis);
    delete[] RotAngles;
    for(int i=0; i<stack->Z(); i++)
	{	if(CmpPlanes[i] == 0L) continue;
		delete[] CmpPlanes[i];
	}
    delete[] CmpPlanes;
    cudaFree(ComRegion);
    cudaFree(CmpPlane);
}

void CalcTIltAxis::CLsetup(float angRange, int num)  //分配了RotAngles和CmpPlanes的内存，要记得释放掉 -
// Allocated memory for RotAngles and CmpPlanes, remember to release it
{
    int ZeroTilt = GetFrameIdxFromTilt(stack->Z(), angles, 0.0f);
    float fTiltAxis = param->rotate[ZeroTilt];  
    float step = angRange / (num - 1);
    RotAngles = new float[num];
    RotAngles[0] = fTiltAxis - step * num / 2;
    for(int i=1; i<num; i++)
    {
        RotAngles[i] = RotAngles[i-1] + step;
    }  
    float fMinSize = (float)1e20;
    for(int i=0; i<num; i++)
    {
        float fsin = (float)fabs(sin(D2R * RotAngles[i]));
        float fcos = (float)fabs(cos(D2R * RotAngles[i]));
        float fSize1 = stack->X() / (fsin + 0.000001f);
		float fSize2 = stack->Y() / (fcos + 0.000001f);
        if(fMinSize > fSize1) fMinSize = fSize1;
		if(fMinSize > fSize2) fMinSize = fSize2;
    }
    LineSize = (int)fMinSize;
    LineSize = LineSize / 2 * 2 - 100;
    CmpLineSize = LineSize / 2 + 1;
    // printf("CommonLine: Line Size = %d\n", LineSize);
	// printf("CommonLine: CmpLineSize = %d\n", CmpLineSize);
    NumLines = num;

    CmpPlanes = new cufftComplex*[stack->Z()];
    int iPixels = num * CmpLineSize;
    for(int i=0; i<stack->Z(); i++)
    {
        CmpPlanes[i] = new cufftComplex[iPixels];
    }
}

void CalcTIltAxis::calcComRegion()
{
    int ZeroTilt = GetFrameIdxFromTilt(stack->Z(), angles, 0.0f);
    float fShift[2] = {0.0f};  //记录图像平移信息 - Record image translation information
    fShift[0] = param->shiftX[ZeroTilt];
    fShift[1] = param->shiftY[ZeroTilt];

    float* gRotAngles;
    size_t iBytes = sizeof(float) * NumLines;
    cudaMalloc(&gRotAngles, iBytes);
    cudaMemcpy(gRotAngles, RotAngles, iBytes, cudaMemcpyDefault);

    dim3 aBlockDim(1, 256);
	dim3 aGridDim(NumLines, 1);
	aGridDim.y = (LineSize + aBlockDim.y - 1) / aBlockDim.y;
    GCalcCommonRegion<<<aGridDim, aBlockDim>>>(stack->X(), stack->Y(), LineSize, gRotAngles,
                                            fShift[0], fShift[1], ComRegion);
	
    cudaFree(gRotAngles);
}

void CalcTIltAxis::Radon(int z)
{
    float fShift[2] = {0.0f};  //平移参数 - translation parameters
    fShift[0] = param->shiftX[z];
    fShift[1] = param->shiftY[z];
    // printf("Radon时的平移参数:%f,%f\n", fShift[0], fShift[1]);
    float fTilt = angles[z];  //倾斜角
    float fcos = (float)cos(fTilt * D2R);

    float *h_proj, *d_proj;   //获取图像 - get image
    int iPixels = stack->X() * stack->Y();
    size_t tBytes= sizeof(float) * iPixels;
    h_proj = new float[iPixels];
    cudaMalloc(&d_proj, tBytes);
    stack->ReadoutputSlice(z, h_proj);
    cudaMemcpy(d_proj, h_proj, tBytes, cudaMemcpyDefault);

    float* gRotAngles;
    int iBytes = sizeof(float) * NumLines;
    cudaMalloc(&gRotAngles, iBytes);
    cudaMemcpy(gRotAngles, RotAngles, iBytes, cudaMemcpyDefault);

    dim3 aBlockDim(1, 512);
	dim3 aGridDim(NumLines, 1);
	aGridDim.y = (LineSize + aBlockDim.y - 1) / aBlockDim.y;
	GRadon<<<aGridDim, aBlockDim>>>(d_proj, stack->X(), stack->Y(), gRotAngles, fcos, 
                        fShift[0], fShift[1], ComRegion, LineSize, (float*)CmpPlane);

    delete[] h_proj;
    cudaFree(d_proj);
    cudaFree(gRotAngles);
}

void CalcTIltAxis::FFT1d()
{
    int PadLineSize = CmpLineSize * 2;
    float* PadLines = (float*)CmpPlane;
    float *PadLine;
    cudaMalloc(&PadLine, sizeof(float) * PadLineSize);
    for(int i=0; i<NumLines; i++)
    {
        cudaMemcpy(PadLine, PadLines + i * PadLineSize, sizeof(float) * PadLineSize, cudaMemcpyDefault);
        dim3 aBlockDim(512, 1);
	    dim3 aGridDim(1, 1);
	    GCalcMean<<<aGridDim, aBlockDim>>>(PadLine, LineSize);

		aGridDim.x = (LineSize + aBlockDim.x -1) / aBlockDim.x;
	    GRemoveMean<<<aGridDim, aBlockDim>>>(PadLine, LineSize);

		m_fft.ForwardFFT(PadLine);
		
        cudaMemcpy(CmpPlane + i * CmpLineSize, (cufftComplex*)PadLine, sizeof(cufftComplex) * CmpLineSize, cudaMemcpyDefault);
    }
    cudaFree(PadLine);  
}

void CalcTIltAxis::CCalcMean(float* PadLine, int LineSize)
{
	float sum = 0.0f;
	int count = 0;
	for(int i=0; i<LineSize; i++)
	{
		if(PadLine[i]>(float)-1e10)
		{
			sum += PadLine[i];
			count++;
		}
	}
	PadLine[LineSize+1] = sum / count;
}

float CalcTIltAxis::findTiltAxis()
{
    int LineMax = 0;
    float fScore = 0.0f;
    float* fScores = new float[NumLines];  //记录每个角度的得分 - record the score of each angle
    // printf("Scores of potential tilt axes.\n");

	for(int i=0; i<NumLines; i++)
    {	//FillLineSet(i);	
        fScores[i] = CalcScore(i);  //计算每个可能的角度的得分 - calculate the score for each possible angle
        if(fScore < fScores[i])   //找到得分最大的角度 - find the angle with the highest score
        {	fScore = fScores[i];
            LineMax = i;
        }
        // printf("...... Score: %4d  %9.5f\n", i, fScores[i]);
    }
    // printf("Best tilt axis: %d, Score: %9.5f\n\n", LineMax, fScore);
    delete[] fScores;
    return RotAngles[LineMax];
}

float CalcTIltAxis::CalcScore(int i)
{   
    cufftComplex *gCmpSum, *gCmpLine, *gCmpRef;
    size_t tBytes = sizeof(cufftComplex) * CmpLineSize;
    cudaMalloc(&gCmpRef, tBytes);
    cudaMalloc(&gCmpLine, tBytes);
    cudaMalloc(&gCmpSum, tBytes);
    cudaMemset(gCmpSum, 0, tBytes);

    float m_fCCSum = 0.0f;
    for(int z=0; z<stack->Z(); z++)
    {
        cufftComplex* gCmpLines = CmpPlanes[z];   //获取第z张图像的所有line - get all lines of the z-th image
        cudaMemcpy(gCmpLine, gCmpLines + i * CmpLineSize, tBytes, cudaMemcpyDefault);   //获取第z张图的第i个角度line - get the i-th angle line of the z-th image
        fftSum(gCmpSum, gCmpLine, 1.0f, 1.0f, gCmpSum, CmpLineSize);
    }

    for(int z=0; z<stack->Z(); z++)
    {
        cufftComplex* m_gCmpLines = CmpPlanes[z];
        cudaMemcpy(gCmpLine, m_gCmpLines + i * CmpLineSize, tBytes, cudaMemcpyDefault); 
        fftSum(gCmpSum, gCmpLine, 1.0f, -1.0f, gCmpRef, CmpLineSize);
        float fCC = GCC1d(gCmpRef, gCmpLine);
        m_fCCSum += fCC;
    }
    m_fCCSum = m_fCCSum / stack->Z();
    cudaFree(gCmpRef);
    cudaFree(gCmpSum);
    cudaFree(gCmpLine);
    return m_fCCSum;
}

void CalcTIltAxis::fftSum(cufftComplex* gCmp1, cufftComplex* gCmp2, float fFact1, float fFact2,
	                cufftComplex* gSum, int iCmpSize)
{
    dim3 aBlockDim(512, 1);
    dim3 aGridDim(1, 1);
    aGridDim.x = iCmpSize / aBlockDim.x + 1;
    GSumCmp<<<aGridDim, aBlockDim>>>(gCmp1, gCmp2, fFact1, fFact2, gSum, iCmpSize);
}

float CalcTIltAxis::GCC1d(cufftComplex* gCmpRef, cufftComplex* gCmpLine)
{
	int iWarps = CalcWarps(CmpLineSize, 32);
    dim3 aBlockDim(iWarps * 32, 1);
	dim3 aGridDim(CmpLineSize / aBlockDim.x + 1, 1);
	int iShmBytes = sizeof(float) * 2 * aBlockDim.x;
	//----------------------------------------------
	int iBlocks = aGridDim.x;
	size_t tBytes = sizeof(float) * iBlocks;  //两倍的线程块个数 - twice the number of thread blocks
	float *gfCC = 0L, *gfStd = 0L;
	cudaMalloc(&gfCC, tBytes);
	cudaMalloc(&gfStd, tBytes);
	//-------------------------
    GConv<<<aGridDim, aBlockDim, iShmBytes>>>
	(gCmpRef, gCmpLine, CmpLineSize, 10, gfCC, gfStd);
	iWarps = CalcWarps(iBlocks, 32);  //对iblocks个数进行归约求和 - Reduce and sum the number of iblocks
	aBlockDim.x = iWarps * 32;
	aGridDim.x = 1;
	iShmBytes = sizeof(float) * aBlockDim.x * 2;
	//------------------------------------------
	GSum<<<aGridDim, aBlockDim, iShmBytes>>>
	(gfCC, gfStd, iBlocks);
    
	float fCC,fStd;
	cudaMemcpy(&fCC, gfCC, sizeof(float), cudaMemcpyDefault);
	cudaMemcpy(&fStd, gfStd, sizeof(float), cudaMemcpyDefault);
	float m_fCC = fCC / fStd;
	cudaFree(gfCC);
	cudaFree(gfStd);

	return m_fCC;
}

int CalcTIltAxis::CalcWarps(int iSize, int iWarpSize)
{
    float fWarps = iSize / (float)iWarpSize;
    if(fWarps < 1.5) return 1;
    //------------------------
    int iExp = (int)(logf(fWarps) / logf(2.0f) + 0.5f);
    int iWarps = 1 << iExp;
    if(iWarps > 16) iWarps = 16;
    return iWarps;
}
