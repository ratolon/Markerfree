#include "CalcReproj.h"
#include "../mrc/mrcstack.h"
#include <stdio.h>
#include <cufft.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>
#include <ctime>
// #include <cuda.h>
// #include <cuda_runtime.h>

#define s_fD2R 0.01745f

static __device__ __constant__ int giProjSize[2];
static __device__ __constant__ int giVolSize[2];

static __device__ float mDIntProj(float* gfProj, float fX)
{
	int x = (int)fX;
	float fVal = gfProj[x];
	if(fVal < (float)-1e10) return (float)-1e30;
	else return fVal;
}

// Code inspired in AreTomo2
static __global__ void mGBackProj
(float* gfSinogram, float* gfTiltAngles, float fProjAngle, int iStartIdx,
	int iEndIdx, float* gfVol // xz slice
)
{	int iX = blockIdx.x * blockDim.x + threadIdx.x;
	if(iX >= giVolSize[0]) return;
	float fX = iX +0.5f - giVolSize[0] * 0.5f;   //fX,fZ为三维体xz切片的坐标 - fX,fZ are the coordinates of the xz slice of the 3D volume
	float fZ = blockIdx.y + 0.5f - giVolSize[1] * 0.5f;
	// float fX = iX - giVolSize[0] * 0.5f;   //fX,fZ为三维体xz切片的坐标 - fX,fZ are the coordinates of the xz slice of the 3D volume
	// float fZ = blockIdx.y - giVolSize[1] * 0.5f;
	//-------------------------------------------------
	float fInt = 0.0f;
	float fCount = 0.0f;
	float fCentX = giProjSize[0] * 0.5f;
	int iEnd = giProjSize[0] - 1;
	//-----------------------------------
	for(int i=iStartIdx; i<=iEndIdx; i++)
	{	float fW = cosf((fProjAngle - gfTiltAngles[i]) * s_fD2R);   //计算权重，角度相差越大，权重越小 - Calculate weight, the greater the angle difference, the smaller the weight
		float fV = gfTiltAngles[i] * s_fD2R;
		float fCos = cosf(fV);
		float fSin = sinf(fV); 
		//--------------------
		fV = fX * fCos + fZ * fSin + fCentX;  //求出正弦图中的ρ - Calculate ρ in the sinogram
		if(fV < 0 || fV > iEnd) continue;     //如果ρ超出正弦图范围则不进行计算 - If ρ exceeds the range of the sinogram, do not calculate
		//-------------------------------
		float* gfProj = gfSinogram + i * giProjSize[0];   //找到正弦图中第i个角度的位置 - Find the position of the i-th angle in the sinogram
		fV = mDIntProj(gfProj, fV);   //取出正弦图中第i个角度，位置为ρ的像素值 - Take out the pixel value of the sinogram at the i-th angle, position ρ
		if(fV < (float)-1e20) continue;
		//------------------------------------------------
		// fV * fCos: distribute the projection intensity
		// along the back-projection ray.
		//------------------------------------------------
		fInt += (fV * fCos * fW);
		fCount += fW;
	}
	int i = blockIdx.y * giVolSize[0] + iX;
	if(fCount < 0.001f) gfVol[i] = (float)-1e30;
	else gfVol[i] = fInt / fCount;
}

extern __shared__ char s_cArray[];

static __global__ void mGForProj
(float* gfVol,int iRayLength,float fCos,
	float fSin,float* gfReproj)
{	float* sfSum = (float*)&s_cArray[0];
	int*  siCount = (int*)&sfSum[blockDim.y];
	sfSum[threadIdx.y] = 0.0f;
	siCount[threadIdx.y] = 0;
	__syncthreads();
	//--------------
	float fXp = blockIdx.x + 0.5f - gridDim.x * 0.5f;
	// float fXp = blockIdx.x - gridDim.x * 0.5f;
	float fTempX = fXp * fCos + giVolSize[0] * 0.5f;   //确定一组在射线上的点，作为起始点 - Determine a set of points on the ray as the starting point
	float fTempZ = fXp * fSin + giVolSize[1] * 0.5f;
	float fZStartp = -fXp * fSin / fCos - 0.5f * iRayLength;	//求出起始点在射线的哪个位置 - Find out where the starting point is on the ray
	//------------------------------------------------------
	int i = 0;
	int iEndX = giVolSize[0] - 1;
	int iEndZ = giVolSize[1] - 1;
	float fX = 0.0f, fZ = 0.0f, fV = 0.0f;
	int iSegments = iRayLength / blockDim.y + 1;
	for(i=0; i<iSegments; i++)
	{	fZ = i * blockDim.y + threadIdx.y;	//fz表示射线上一个点 - fz represents a point on the ray
		if(fZ >= iRayLength) continue;
		//----------------------------
		fZ = fZ + fZStartp;			//求出这一点和初始点的相对位置 - Find the relative position of this point and the starting point
		fX = fTempX - fZ * fSin;	//求出对应的图像坐标 - Find the corresponding image coordinates
		fZ = fTempZ + fZ * fCos;
		//----------------------
		if(fX >= 0 && fX < iEndX && fZ >= 0 && fZ < iEndZ)
		{	fV = gfVol[giVolSize[0] * (int)(fZ) + (int)(fX)];
			if(fV >= (float)-1e10)
			{	sfSum[threadIdx.y] += fV;
				siCount[threadIdx.y] += 1;
			}
		}
	}
	__syncthreads();
	//--------------
	i = blockDim.y / 2;
	while(i > 0)
	{	if(threadIdx.y < i)
		{	sfSum[threadIdx.y] += sfSum[threadIdx.y+i];
			siCount[threadIdx.y] += siCount[threadIdx.y+i];
		}
		__syncthreads();
		i /= 2;
	}
	//-------------
	if(threadIdx.y != 0) return;
	if(siCount[0] < 0.8f * iRayLength) 		//如果射线在图像内长度小于最大长度的0.8倍，则被放弃，重投影值设为负大值 
	// - If the length of the ray inside the image is less than 0.8 times the maximum length
	{	gfReproj[blockIdx.x] = (float)-1e30;
	}
	else 
	{	gfReproj[blockIdx.x] = sfSum[0] / siCount[0];	//结果归一化 - Result normalization
	}
}

CalcReprojFBP::CalcReprojFBP()
{
	fSinogram = 0L;
	TiltAngles = 0L;
	gReproj = 0L;
	fvol = 0L;
}

CalcReprojFBP::~CalcReprojFBP()
{
	this->Clean();
}

void CalcReprojFBP::Clean()
{
	if(fSinogram != 0L) cudaFree(fSinogram);  //一张投影图像的xz切片，大小为x*z - An xz slice of a projection image, size x*z
	if(TiltAngles != 0L) cudaFree(TiltAngles); //倾斜角数据，大小为z - Tilt angle data, size z
	if(gReproj != 0L) cudaFree(gReproj);      //重投影过程中使用的中间变量 - Intermediate variable used in the reprojection process
	if(fvol != 0L) cudaFree(fvol);           //重建三维体的xz切片，大小为x*thickness - xz slice of the reconstructed 3D volume, size x*thickness
	fSinogram = 0L;
	TiltAngles = 0L;
	gReproj = 0L;
	fvol = 0L;
	// m_greproj.Clean();
}

void CalcReprojFBP::Setup(int* rprojsize, int ivolZ, int num, std::vector<float> s_angles)
{
    memcpy(projsize, rprojsize, sizeof(int) * 2);
    numprojs = num;
    angles = s_angles;
    int iSinoPixels = projsize[0] * numprojs;
	size_t tSinoBytes = iSinoPixels * sizeof(float);
    cudaMalloc(&fSinogram, tSinoBytes);
	cudaMalloc(&TiltAngles, numprojs * sizeof(float));
	cudaMemcpy(TiltAngles, angles.data(), numprojs * sizeof(float), cudaMemcpyHostToDevice);
	// int iVolX = 2 * projsize[0];
	// m_greproj.setsize(projsize[0], numprojs, iVolX, ivolZ);
    m_aiProjSize[0] = projsize[0];
	m_aiProjSize[1] = numprojs;
    int iVolX = 2 * projsize[0];
    m_aiVolSize[0] = iVolX;
	m_aiVolSize[1] = ivolZ;
    int iBytes = sizeof(int) * 2;
	cudaMemcpyToSymbol(giProjSize, m_aiProjSize, iBytes);
	cudaMemcpyToSymbol(giVolSize, m_aiVolSize, iBytes);

	iBytes = m_aiProjSize[0] * sizeof(float);
	cudaMalloc(&gReproj, iBytes);
	//------------------------------
	iBytes = m_aiVolSize[0] * m_aiVolSize[1] * sizeof(float);
	cudaMalloc(&fvol, iBytes);    //一张体的xz方向切片 - An xz slice of the volume
}

void CalcReprojFBP::DoIt(float* corrprojs, bool* SkipProjs, int iProj, float* fReproj)
{
	m_corrprojs = corrprojs;
	m_fReproj = fReproj;
	m_iProj = iProj;
	// clock_t start_time = clock();
	FindProjRange(angles, SkipProjs);
	// size_t tBytes = sizeof(float) * numprojs;
	// cudaMemcpy(TiltAngles, angles.data(), tBytes, cudaMemcpyHostToDevice);
	// clock_t part1_end_time = clock();
	for(int y=0; y < projsize[1]; y++)
	{
		GetSinogram(y);
		Reproj(y, angles[m_iProj]);
	}
	// clock_t end_time = clock();
	// double part1_duration = double(part1_end_time - start_time) / CLOCKS_PER_SEC * 1000;
    // double part2_duration = double(end_time - part1_end_time) / CLOCKS_PER_SEC * 1000;
    // double total_duration = double(end_time - start_time) / CLOCKS_PER_SEC * 1000;

    // // // 输出各部分的执行时间
    // std::cout << "FindProjRange时间：" << part1_duration << " 毫秒" << std::endl;
    // std::cout << "Reproj时间：" << part2_duration << " 毫秒" << std::endl;
    // std::cout << "总执行时间：" << total_duration << " 毫秒" << std::endl;
}

void CalcReprojFBP::FindProjRange(std::vector<float> angles, bool* SkipProjs)
{
	float RefRange = 20.5f; //20.5f;
    float RefStretch = 1.2f; //(float)(1.0 / cos(40.0 * D2R));	
    float Tilt = angles[m_iProj];
    float fcos = (float)cos(Tilt * D2R);
    
    int istart = -1;
    for(int i=0; i<numprojs; i++)
    {
        if(SkipProjs[i]) continue;

        float fTiltA = angles[i];
        float fDiffA = Tilt - fTiltA;
        if(fabs(fDiffA) > RefRange) continue;

        float fStretch = (float)(cos(fTiltA * D2R) / fcos);
        if(fStretch > RefStretch) continue;

        istart = i;
        break;
    }//找到第一个符合要求的序号 - Find the first serial number that meets the requirements
    int iend = -1;
    for(int i=istart; i<numprojs; i++)
	{	if(SkipProjs[i]) continue;
		//--------------------------
		float fTiltA = angles[i];
		float fDiffA = Tilt - fTiltA;
		if(fabs(fDiffA) > RefRange) continue;
		//------------------------------------
		float fStretch = (float)(cos(fTiltA * D2R) / fcos);
		if(fStretch > RefStretch) continue;
		//----------------------------------
		iend = i;
	}//找到最后一个符合要求的序号 - Find the last serial number that meets the requirements
	// if((iend - istart) > 9) iend = istart + 9;
	int nproj = nProj - 1;
	// std::cout << "nProj:" << nproj << std::endl;
	if ((iend - istart) > nproj)
	{
		int iZeroTilt = GetFrameIdxFromTilt(numprojs, angles, 0.0f);
		if(m_iProj < iZeroTilt)
			iend = istart + nproj;
		else
			istart = iend - nproj;
	}
	ProjRange[0] = istart;
	ProjRange[1] = iend;
	
	// printf("%.2f  %d  %d  %d  %.2f  %.2f\n", angles[m_iProj],
    //        ProjRange[0], ProjRange[1],
    //        ProjRange[1] - ProjRange[0] + 1,
	//    angles[ProjRange[0]], angles[ProjRange[1]]);
}

void CalcReprojFBP::GetSinogram(int y)
{
    size_t tPixels = projsize[0] * projsize[1];
	size_t tLineBytes = sizeof(float) * projsize[0];
	//size_t tSinoBytes = tLineBytes * stack->Z();
	int iOffsetY = y * projsize[0];
	//----------------------------------
	for(int i=ProjRange[0]; i<=ProjRange[1]; i++)
	{	float* pfProj = m_corrprojs + i * tPixels;
		float* pfSrc = pfProj + iOffsetY;
		float* gfDst = fSinogram + i * projsize[0];
		cudaMemcpy(gfDst, pfSrc, tLineBytes, cudaMemcpyDefault);
	}
}

void CalcReprojFBP::Reproj(int y, float projangle)
{
	BackProj(projangle);
	ForwardProj(projangle);
	// m_greproj.DoIt(fSinogram, TiltAngles, ProjRange, projangle);
	
	size_t tLineBytes = sizeof(float) * projsize[0];
	float* fLine = m_fReproj + y * projsize[0];
	cudaMemcpy(fLine, gReproj, tLineBytes, cudaMemcpyDefault); 
	// cudaMemcpy(fLine, m_greproj.gReproj, tLineBytes, cudaMemcpyDefault); 
}

void CalcReprojFBP::BackProj(float projangle)
{
	dim3 aBlockDim(512, 1);
	dim3 aGridDim(1, m_aiVolSize[1]);
	aGridDim.x = m_aiVolSize[0] / aBlockDim.x + 1;
	//--------------------------------------------
	mGBackProj<<<aGridDim, aBlockDim, 0, m_stream>>>
	(fSinogram, TiltAngles, projangle, 
	  ProjRange[0], ProjRange[1], fvol);
}

void CalcReprojFBP::ForwardProj(float projangle)
{
	float fCos = (float)cos(D2R * projangle);
	float fSin = (float)sin(D2R * projangle);
	int iRayLength = (int)(m_aiVolSize[1] / fCos + 0.5f);
        //---------------------------------------------------
	dim3 aBlockDim(1, 512);
	dim3 aGridDim(m_aiProjSize[0], 1);
	int iShmBytes = (sizeof(float) + sizeof(int)) * aBlockDim.y;
	//----------------------------------------------------------
	mGForProj<<<aGridDim, aBlockDim, iShmBytes, m_stream>>>
	(fvol, iRayLength, fCos, fSin, gReproj);
}
