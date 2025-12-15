#include "CorrStack.h"
//----------------------------------------------------------------------
void CorrUtil::CalcAlignedSize
(int* piRawSize, float fTiltAxis,int* piAlnSize)
{	memcpy(piAlnSize, piRawSize, sizeof(int) * 2);
	double dRot = fabs(sin(fTiltAxis * 3.14 / 180.0));
	// if(dRot <= 0.707) return;
	if(dRot <= 1) return;
	//-----------------------
	piAlnSize[0] = piRawSize[1];
	piAlnSize[1] = piRawSize[0];
	// piAlnSize[0] = piRawSize[0];
	// piAlnSize[1] = piRawSize[1];
}

void CorrUtil::CalcBinnedSize
(int* piRawSize, float fBinning, int* piBinnedSize)
{	
	int iBin = (int)(fBinning + 0.5f);
	bool bPadded = true;
	BinImage::GetBinSize(piRawSize, !bPadded,
		   iBin, piBinnedSize, !bPadded);
}

void CorrUtil::Unpad(float* pfPad, int* piPadSize, float* pfImg)
{
	int iImageX = (piPadSize[0] / 2 - 1) * 2;
	int iBytes = iImageX * sizeof(float);
	//-----------------------------------
	for(int y=0; y<piPadSize[1]; y++)
	{	float* pfSrc = pfPad + y * piPadSize[0];
		float* pfDst = pfImg + y * iImageX;
		cudaMemcpy(pfDst, pfSrc, iBytes, cudaMemcpyDefault);
	}
}
//----------------------------------------------------------------------
static __device__ __constant__ int giBinInSize[2];
static __device__ __constant__ int giBinOutSize[2];

static __global__ void mGBinImage
(	float* gfInImg,
	int iBinX,
	int iBinY,
	float* gfOutImg
)
{	int y =  blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= giBinOutSize[1]) return;
	int i = y * giBinOutSize[0] + blockIdx.x;
	gfOutImg[i] = (float)-1e30;
	//-------------------------
	int x =  blockIdx.x * iBinX;
	y = y * iBinY;
	float fSum = 0.0f;
	for(int iy=0; iy<iBinY; iy++)
	{	float* pfPtr = gfInImg + (y + iy) * giBinInSize[0];
		for(int ix=0; ix<iBinX; ix++)
		{	float fVal = pfPtr[x + ix];
			if(fVal < (float)-1e10) return;
			else fSum += fVal;
		}
	}
	gfOutImg[i] = fSum;
}

void BinImage::GetBinSize
(	int* piInSize, bool bInPadded, int iBinning,
	int* piOutSize, bool bOutPadded
)
{	int iImgX = piInSize[0];
	if(bInPadded) iImgX = (piInSize[0] / 2 - 1) * 2;
	//----------------------------------------------
	piOutSize[0] = iImgX / iBinning / 2 * 2;
	piOutSize[1] = piInSize[1] / iBinning / 2 * 2;
	if(bOutPadded) piOutSize[0] += 2;
}

BinImage::BinImage(void)
{
}

BinImage::~BinImage(void)
{
}

void BinImage::SetupSizes
(	int* piInSize, bool bInPadded,
	int* piOutSize, bool bOutPadded
)
{	int iBytes = sizeof(int) * 2;
	cudaMemcpyToSymbol(giBinInSize, piInSize, iBytes);
	cudaMemcpyToSymbol(giBinOutSize, piOutSize, iBytes);
	memcpy(m_aiOutSize, piOutSize, iBytes);
	//-------------------------------------
	int iInImgX = piInSize[0];
	m_iOutImgX = piOutSize[0];
	if(bInPadded) iInImgX = (piInSize[0] / 2 - 1) * 2;
	if(bOutPadded) m_iOutImgX = (piOutSize[0] / 2 - 1) * 2;
	//-----------------------------------------------------
	m_aiBinning[0] = iInImgX / m_iOutImgX;
	m_aiBinning[1] = piInSize[1] / piOutSize[1];
}

void BinImage::DoIt
(float* gfInImg, float* gfOutImg)
{	dim3 aBlockDim(1, 512);
	dim3 aGridDim(m_iOutImgX, 1);
	aGridDim.y = (m_aiOutSize[1] + aBlockDim.y - 1) / aBlockDim.y;
	//----------------------------------------------------------
	mGBinImage<<<aGridDim, aBlockDim>>>(gfInImg, 
	   m_aiBinning[0], m_aiBinning[1], gfOutImg);
}
//----------------------------------------------------------------------
static __device__ __constant__ int giInSize[2];
static __device__ __constant__ int giOutSize[2];

static __device__ float mGRandom
(	int x, int y, 
	int iInImgX,
	float* gfInImg
)
{	if(x < 0) x = -x;
	if(y < 0) y = -y;
	if(x >= iInImgX) x = iInImgX - 1 - (x % iInImgX);
	if(y >= giInSize[1]) y = giInSize[1] - 1 - (y % giInSize[1]);
	//-----------------------------------------------------------
	int iWin = 51, ix = 0, iy = 0;
	int iSize = iWin * iWin;
	unsigned int next = y * giInSize[0] + x;
	for(int i=0; i<20; i++)
	{	next = (next * 19 + 57) % iSize;
		ix = (next % iSize) - iWin / 2 + x;
		if(ix < 0 || ix >= iInImgX) continue;
		//-----------------------------------
		iy = (next / iWin) - iWin / 2 + y;
		if(iy < 0 || iy >= giInSize[1]) continue;
		//---------------------------------------
		return gfInImg[iy * giInSize[0] + ix];
	}
	return gfInImg[y * giInSize[0] + x];
}

static __global__ void mGCorrect
(	float* gfInImg,
	int iInImgX,
	float fGlobalShiftX,
	float fGlobalShiftY,
	float fRotAngle, // tilt axis in radian
	bool bRandomFill,
	float* gfOutImg
)
{	int x = 0, y = 0;
	y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= giOutSize[1]) return;
	int i = y * giOutSize[0] + blockIdx.x;
	//------------------------------------
	float afXY[2] = {0.0f}, afTmp[2];
	afXY[0] = blockIdx.x + 0.5f - gridDim.x * 0.5f;
	afXY[1] = y + 0.5f - giOutSize[1] * 0.5f;
	//---------------------------------------
	afTmp[0] = cosf(fRotAngle);
	afTmp[1] = sinf(fRotAngle);
	float fT = afXY[0] * afTmp[0] - afXY[1] * afTmp[1]; 
	afXY[1] = afXY[0] * afTmp[1] + afXY[1] * afTmp[0];
	afXY[0] = fT;
	//-----------
	afXY[0] += (fGlobalShiftX + iInImgX * 0.5f);
	afXY[1] += (fGlobalShiftY + giInSize[1] * 0.5f);
	//----------------------------------------------
	x = (int)(afXY[0] + 0.5f);
	y = (int)(afXY[1] + 0.5f);
	//------------------------
	if(x >= 0 && x < iInImgX && y >= 0 && y < giInSize[1]) 
	{	gfOutImg[i] = gfInImg[y * giInSize[0] + x];
		return;
	}
	//-------------
	if(bRandomFill) gfOutImg[i] = mGRandom(x, y, iInImgX, gfInImg);
	else gfOutImg[i] = (float)(-1e30);
}

GCorrPatchShift::GCorrPatchShift(void)
{
}

GCorrPatchShift::~GCorrPatchShift(void)
{
}

void GCorrPatchShift::SetSizes
(	int* piInSize,
	bool bInPadded, 
	int* piOutSize,
	bool bOutPadded
)
{	int aiInSize[] = {piInSize[0], piInSize[1]};
	cudaMemcpyToSymbol(giInSize, aiInSize, sizeof(giInSize));
	cudaMemcpyToSymbol(giOutSize, piOutSize, sizeof(giOutSize));
	//----------------------------------------------------------
	m_iInImgX = piInSize[0];
	if(bInPadded) m_iInImgX = (piInSize[0] / 2 - 1) * 2;
	//--------------------------------------------------
	m_iOutImgY = piOutSize[1];
	m_iOutImgX = piOutSize[0];
	if(bOutPadded) m_iOutImgX = (piOutSize[0] / 2 - 1) * 2;
}

void GCorrPatchShift::DoIt
(	float* gfInImg,
	float* pfGlobalShift,
	float fRotAngle,
	bool bRandomFill,
	float* gfOutImg
)
{	
	dim3 aBlockDim(1, 512);
	dim3 aGridDim(m_iOutImgX, 1);
	aGridDim.y = (m_iOutImgY + aBlockDim.y - 1) / aBlockDim.y;
	//--------------------------------------------------------
	mGCorrect<<<aGridDim, aBlockDim>>>(gfInImg, m_iInImgX,
	   pfGlobalShift[0], pfGlobalShift[1], fRotAngle, bRandomFill, gfOutImg);
}
//----------------------------------------------------------------------
static float* sAllocGPU(int* piSize, bool bPad)
{
	float* gfBuf = 0L;
	int iSizeX = bPad ? (piSize[0] / 2 + 1) * 2 : piSize[0];
	size_t tBytes = sizeof(float) * iSizeX * piSize[1];
	cudaMalloc(&gfBuf, tBytes);
	return gfBuf;
}

static void sCalcPadSize(int* piSize, int* piPadSize)
{
	piPadSize[0] = (piSize[0] / 2 + 1) * 2;
	piPadSize[1] = piSize[1];
}

// static void sCalcCmpSize(int* piSize, int* piCmpSize)
// {
// 	piCmpSize[0] = piSize[0] / 2 + 1;
// 	piCmpSize[1] = piSize[1];
// }

CorrTomoStack::CorrTomoStack()
{
	m_gfRawProj = 0L;
	m_gfCorrProj = 0L;
	m_gfBinProj = 0L;
	m_gfCorrectProjs = 0L;
	m_pRrweight = 0L;
	h_proj = 0L;
}

CorrTomoStack::~CorrTomoStack()
{
	this->Clean();
}

void CorrTomoStack::Clean()
{
	if(m_gfRawProj != 0L) cudaFree(m_gfRawProj);
	if(m_gfCorrProj != 0L) cudaFree(m_gfCorrProj);
	if(m_gfBinProj != 0L) cudaFree(m_gfBinProj);
	if(m_gfCorrectProjs != 0L) cudaFree(m_gfCorrectProjs);
	if(m_pRrweight != 0L) delete m_pRrweight;
	if(h_proj != 0L) delete[] h_proj;
	m_gfRawProj = 0L;
	m_gfCorrProj = 0L;
	m_gfBinProj = 0L;
	m_gfCorrectProjs = 0L;
	m_pRrweight = 0L;
	h_proj = 0L;
}

void CorrTomoStack::GetBinning(int* pfBinning)
{
	pfBinning[0] = m_aiBinnedSize[0];
	pfBinning[1] = m_aiBinnedSize[1];
}

float* CorrTomoStack::GetCorrectedProjs()
{
	return m_gfCorrectProjs;
}

void CorrTomoStack::Set0(MrcStackM* stack, AlignParam* param)
{
    mstack = stack;
    mparam = param;
	m_aiStkSize[0] = mstack->X();
	m_aiStkSize[1] = mstack->Y();
	m_aiStkSize[2] = mstack->Z();
	size_t iPixels = m_aiStkSize[0] * m_aiStkSize[1];
	h_proj = new float[iPixels];
	// printf("m_aiStkSize:%d, %d, %d\n", m_aiStkSize[0], m_aiStkSize[1], m_aiStkSize[2]);
}

void CorrTomoStack::Set1(float fTiltAxis, float fOutBin)
{
    CorrUtil::CalcAlignedSize(m_aiStkSize, fTiltAxis, m_aiAlnSize);
    m_aiAlnSize[2] = m_aiStkSize[2];

    bool bPad = true, bPadded = true;
	m_gfRawProj = sAllocGPU(m_aiStkSize, !bPad);
    int* piCorrSize = m_aiAlnSize;
	if(m_aiAlnSize[1] < m_aiStkSize[1]) piCorrSize = m_aiStkSize;
	m_gfCorrProj = sAllocGPU(piCorrSize, bPad);

    int aiAlnPadSize[2] = {0};
    sCalcPadSize(m_aiAlnSize, aiAlnPadSize);
    m_aGCorrPatchShift.SetSizes(m_aiStkSize, !bPadded, aiAlnPadSize, bPadded);
	
	//---------------------------------------------------------------
	m_fOutBin = fOutBin;
	CorrUtil::CalcBinnedSize(m_aiAlnSize, m_fOutBin, m_aiBinnedSize);
	m_aiBinnedSize[2] = m_aiStkSize[2];

	m_afBinning[0] = m_aiAlnSize[0] / (float)m_aiBinnedSize[0];
	m_afBinning[1] = m_aiAlnSize[1] / (float)m_aiBinnedSize[1];

	m_gfBinProj = sAllocGPU(m_aiBinnedSize, bPad);
	size_t tBytes = m_aiBinnedSize[0] * m_aiBinnedSize[1] * m_aiBinnedSize[2] * sizeof(float);
	cudaMalloc(&m_gfCorrectProjs, tBytes);

	int aiBinPadSize[2];
	sCalcPadSize(m_aiBinnedSize, aiBinPadSize);
	m_aGBinImg2D.SetupSizes(aiAlnPadSize, bPadded, aiBinPadSize, bPadded);
}

void CorrTomoStack::Set2(bool bRandFill, bool bShiftOnly, bool bRweight)
{
	m_bRandomFill = bRandFill;

	m_onlyshift = bShiftOnly;

	int aiBinPadSize[2], aiAlnPadSize[2];
	sCalcPadSize(m_aiAlnSize, aiAlnPadSize);
	sCalcPadSize(m_aiBinnedSize, aiBinPadSize);
	if(bRweight)
	{
		RWeight* pGRweight = new RWeight;
		pGRweight->Setup(aiBinPadSize[0], aiBinPadSize[1]);
		m_pRrweight = pGRweight;
	}
}

void CorrTomoStack::Set3(MrcStackM* stack, bool write)
{
	m_bWrite = write;
	if(m_bWrite) m_WriteFile = stack;
}

void CorrTomoStack::DoIt()
{
	size_t tpxsize = m_aiBinnedSize[0] * m_aiBinnedSize[1];
	size_t tBytes = tpxsize * sizeof(float);
	if(m_bWrite) h_newproj = new float[tpxsize];  
	for(int i=0; i<m_aiStkSize[2]; i++)
	{
		mCorrectProj(i);      
        if(m_bWrite)
        {
            cudaMemcpy(h_newproj, m_gfCorrectProjs + i*tpxsize, tBytes, cudaMemcpyDefault);
            m_WriteFile->WriteSlice(i, h_newproj); 
        }  
	}
	if(m_bWrite) delete[] h_newproj;
}

void CorrTomoStack::mCorrectProj(int iProj)
{
	float afShift[2] = {0.0f};
	afShift[0] = mparam->shiftX[iProj];
    afShift[1] = mparam->shiftY[iProj];

	float fTiltAxis = mparam->rotate[iProj];  
    if(m_onlyshift) fTiltAxis = 0;
	fTiltAxis *= D2R;

	size_t iPixels = m_aiStkSize[0] * m_aiStkSize[1];
	size_t tBytes = sizeof(float) * iPixels;
	mstack->ReadoutputSlice(iProj, h_proj);
    cudaMemcpy(m_gfRawProj, h_proj, tBytes, cudaMemcpyDefault); 
	m_aGCorrPatchShift.DoIt(m_gfRawProj, afShift, fTiltAxis, 
		m_bRandomFill, m_gfCorrProj);

	m_aGBinImg2D.DoIt(m_gfCorrProj, m_gfBinProj);

	if(m_pRrweight != 0L)
	{	
		m_pRrweight->DoIt(m_gfBinProj);
	}
	int aiPadSize[] = {0, m_aiBinnedSize[1]};
	aiPadSize[0] = (m_aiBinnedSize[0] / 2 + 1) * 2;
	float* pfProjOut = m_gfCorrectProjs + iProj * m_aiBinnedSize[0] * m_aiBinnedSize[1];
	CorrUtil::Unpad(m_gfBinProj, aiPadSize, pfProjOut);
}

RWeight::RWeight()
{
    pGForward = 0L;
    pGInverse = 0L;
}

RWeight::~RWeight()
{
    if(pGForward != 0L) delete pGForward;
	if(pGInverse != 0L) delete pGInverse;
	pGForward = 0L;
	pGInverse = 0L;
    // forward.DestroyPlan();
    // inverse.DestroyPlan();
}

void RWeight::Setup(int iPadProjX, int iNumProjs) 
{
	int iFFTSize = (iPadProjX / 2 - 1) * 2;
	CmpSize[0] = iFFTSize / 2 + 1;
	CmpSize[1] = iNumProjs;
    printf("cmpsize:%d,%d\n", CmpSize[0], CmpSize[1]);
	//----------------------
    pGForward = new cufft1D;
	pGInverse = new cufft1D;
    bool bForward = true;
	pGForward->CreatePlan(iFFTSize,CmpSize[1],iPadProjX,bForward);
    pGInverse->CreatePlan(iFFTSize,CmpSize[1],iPadProjX,!bForward);
    // forward.CreateForwardPlan(CmpSize);
    // inverse.CreateInversePlan(CmpSize);
}

__global__ void GRWeight(cufftComplex* gCmpSinogram, int iCmpSize)
{	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x >= iCmpSize) return;
	int i = blockIdx.y * iCmpSize + x;
	//--------------------------------
	float fN = 2 * (iCmpSize - 1.0f);
	gCmpSinogram[i].x /= fN;
	gCmpSinogram[i].y /= fN;
	//----------------------
	float fR = x / fN;
	fR = 2 * fR * (0.55f + 0.45f * cosf(6.2831852f * fR));  //这里使用的公式和书上的不太一样，后续改改试试 
	// - The formula used here is a bit different from that in the book, try to change it later
    // fR = 2 * fR * (0.54f - 0.46f * cosf(6.2831852f * fR));
	gCmpSinogram[i].x *= fR;
	gCmpSinogram[i].y *= fR;
}

void RWeight::DoIt(float* gfPadSinogram)
{	
    // bool bNorm = true;
	// forward.ForwardFFT(gfPadSinogram, CmpSize, !bNorm);
	pGForward->Forward(gfPadSinogram);
	cufftComplex* gCmpSinogram = (cufftComplex*)gfPadSinogram;
	//-------------------------------------------------------
	dim3 aBlockDim(64, 1);
	dim3 aGridDim(1, CmpSize[1]);
	aGridDim.x = (CmpSize[0] + aBlockDim.x - 1) / aBlockDim.x;
	GRWeight<<<aGridDim, aBlockDim>>>(gCmpSinogram, CmpSize[0]);
	//------------------------------------------------------------
	pGInverse->Inverse(gCmpSinogram);
    // inverse.InverseFFT(gCmpSinogram);
}
