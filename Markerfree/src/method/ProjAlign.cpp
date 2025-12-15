#include "ProjAlign.h"
#include "../mrc/mrcstack.h"
#include <stdio.h>
#include <cufft.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>

ProjAlign::ProjAlign()
{
	m_corrprojs = 0L;
	SkipProjs = 0L;
	fReproj = 0L;
	// mGreproj = 0L;
	PadRef = 0L;
	PadImg = 0L;
	m_pCorrTomoStack = 0L;
}

ProjAlign::~ProjAlign()
{
    this->Clean();
}

void ProjAlign::Clean()
{
	if(m_corrprojs != 0L) cudaFree(m_corrprojs);	  //根据已有参数矫正图像部分，删除的时候也会释放m_corr内存 - Release m_corr memory when deleting according to existing parameters to correct the image part
    if(SkipProjs != 0L) delete[] SkipProjs;   //记录已经纠正过的单张投影，占内存很小 - Records a single projection that has been corrected, occupies very little memory
    if(fReproj != 0L) cudaFree(fReproj);      //一张重投影，大小为x*y - A reprojected image, size x*y
	// if(mGreproj != 0L) cudaFree(mGreproj);
	if(PadRef != 0L) cudaFree(PadRef);
	if(PadImg != 0L) cudaFree(PadImg);
	if(m_pCorrTomoStack != 0L) delete m_pCorrTomoStack;
	m_pCorrTomoStack = 0L;
	m_corrprojs = 0L;
	SkipProjs = 0L;
	fReproj = 0L;
	// mGreproj = 0L;
	PadRef = 0L;
	PadImg = 0L;
	m_fft.DestroyPlan();
    in_fft.DestroyPlan();
}

void ProjAlign::Setup(MrcStackM &preprojs, AlignParam* pAlignParam, std::vector<float> p_angles, float zoffset, int thick)
{
    stack = &preprojs;
    param = pAlignParam;
    angles = p_angles;
    m_fZ0 = zoffset;
	m_fX0 = 0;
    thickness = thick;

	float XcfSize = 2048.0f;
    int BinX = (int)(stack->X() / XcfSize + 0.5f);
    int BinY = (int)(stack->Y() / XcfSize + 0.5f);
    iBin = (BinX > BinY) ? BinX : BinY;
    if(iBin<1) iBin = 1;
	// if(iBin%2) iBin = iBin / 2 * 2;
	bool exchange = true;
	// bool exchange = false;
	Setsize(!exchange);	
}

void ProjAlign::Setsize(bool exchange)
{
	this->Clean();
	int bin = iBin;
	printf("bin:%d   ", bin);
	// corrstack = new CorrStack;
    // corrstack->stack = stack;
    // corrstack->param = param; 
	// float fTiltAxis = param->rotate[stack->Z() / 2];
	// // printf("fTiltAxis:%f\n", fTiltAxis);
	// if(exchange) {
	// 	corrstack->SetSize(fTiltAxis, bin);
	// 	printf("以下交换长宽\n");
	// }
	// else {
	// 	corrstack->SetSize(0, bin);
	// 	printf("以下不交换长宽\n");
	// }
	// bool onlyshift = true; bool randomfill = true; 
	// bool write = true;     bool rweight = true; 
	// corrstack->Setup(!onlyshift, randomfill, !write, rweight);  //rweight改成
	// m_corrprojs = corrstack->corrprojs;

	float fTiltAxis = param->rotate[stack->Z() / 2];
	bool onlyshift = true; bool randomfill = true; 
	bool write = true;     bool rweight = true; 
	m_pCorrTomoStack = new CorrTomoStack;
	m_pCorrTomoStack->Set0(stack, param);
	m_pCorrTomoStack->Set1(fTiltAxis, bin);
	m_pCorrTomoStack->Set2(randomfill, !onlyshift, rweight);
	m_pCorrTomoStack->Set3(0L, !write);
	m_corrprojs = m_pCorrTomoStack->GetCorrectedProjs();
	m_pCorrTomoStack->GetBinning(BinSize);
    
    // BinSize[0] = corrstack->BinSize[0];
    // BinSize[1] = corrstack->BinSize[1];
    CentSize[0] = BinSize[0];
	CentSize[1] = BinSize[1];
	PadBinSize[0] = (CentSize[0] / 2 + 1) * 2;
	PadBinSize[1] = CentSize[1];
	CmpBinSize[0] = PadBinSize[0] / 2;
	CmpBinSize[1] = PadBinSize[1];
	printf("binsize:%d, %d\n", BinSize[0], BinSize[1]);

    m_thickness = thickness / bin / 2 * 2;
	size_t tBytes = PadBinSize[0] * PadBinSize[1] * sizeof(float);
    cudaMalloc(&fReproj, sizeof(float) * BinSize[0] * BinSize[1]);  
	cudaMalloc(&PadRef, tBytes);
	cudaMalloc(&PadImg, tBytes);   
	// cudaMalloc(&mGreproj, sizeof(float) * BinSize[0] * BinSize[1] * stack->Z());  //为pnp方法准备,搜索所有pnp注释掉
	SkipProjs = new bool[stack->Z()];
    gcc.SetSize(BinSize);

	m_calcreproj.Setup(BinSize, m_thickness, stack->Z(), angles);

	m_fft.CreateForwardPlan(CmpBinSize);
	in_fft.CreateInversePlan(CmpBinSize);
}

float ProjAlign::DoIt()
{   
	float fMaxErr = DoItBin();  
	return fMaxErr;
}

float ProjAlign::DoItBin()
{
	// corrstack->CorrectProjs();
	m_pCorrTomoStack->DoIt();
	
	FitRotCenterZ();
	RemoveOffsetZ(-1.0f);

	// setreproj();
	// int iPixels = BinSize[0] * BinSize[1];
	// size_t tybes = iPixels * sizeof(float);
	// for(int i=0;i<stack->Z(); i++)
	// {
	// 	cudaMemcpy(h_data, m_corrprojs+i*iPixels, tybes, cudaMemcpyDefault);
	// 	reproj.WriteSlice(i, h_data);
	// }
	// closereproj();

	int ZeroTilt = GetFrameIdxFromTilt(stack->Z(), angles, 0.0f);
	for(int i=0; i<stack->Z(); i++)
	{
		SkipProjs[i] = true;
	}
	SkipProjs[ZeroTilt] = false;

	printf("# Projection matching measurements\n");
	printf("# tilt angle  x shift  y shift\n");
    float fMaxErr = (float)-1e20;
	for(int i=1; i<stack->Z(); i++)
	{	int iProj; 
		iProj= ZeroTilt + i;
		if(iProj < stack->Z() && iProj >= 0)
		{	float fErr = ClacAlignProj(iProj);
			SkipProjs[iProj] = false;
			if(fErr > fMaxErr) fMaxErr = fErr;
		}
		//----------------------------------------
		iProj = ZeroTilt - i;
		if(iProj < stack->Z() && iProj >= 0)
		{	float fErr = ClacAlignProj(iProj);
			SkipProjs[iProj] = false;
			if(fErr > fMaxErr) fMaxErr = fErr;
		}
	}
	// closereproj();
	return fMaxErr;
}

float ProjAlign::ClacAlignProj(int iproj, bool zero)
{
	// clock_t start_time = clock();   //删除的时候检索clock_t关键词即可，还有最后一段计算和输出时间的部分

	m_calcreproj.DoIt(m_corrprojs, SkipProjs, iproj, fReproj);
	// int iPixels = BinSize[0] * BinSize[1];  //这两步为pnp准备
	// cudaMemcpy(fReproj, mGreproj + iproj * iPixels, iPixels * sizeof(float), cudaMemcpyDefault);    //这两步为pnp准备
	// mpnp_calcreproj.DoIt(fReproj, angles, iproj, SkipProjs);
	// clock_t part1_end_time = clock();

	// int iPixels = BinSize[0] * BinSize[1];
	// size_t tybes = iPixels * sizeof(float);
	// cudaMemcpy(h_data, fReproj, tybes, cudaMemcpyDefault);
	// reproj.WriteSlice(iproj, h_data);

	MeaAlignProj(iproj);

	float afShift[2] = {0.0f};
	float binX = stack->X() / (float)BinSize[0];
	float binY = stack->Y() / (float)BinSize[1];
	afShift[0] = ishiftX * binX;
	afShift[1] = ishiftY * binY;
	printf("  %6.2f  %8.2f  %8.2f\n", angles[iproj], afShift[0], afShift[1]);
	float fShift = afShift[0] * afShift[0] + afShift[1] * afShift[1]; 
	fShift = (float)sqrt(fShift);
	//---------------------------
	float fTiltAxis = param->rotate[iproj];	
	RotShift(afShift, fTiltAxis, afShift);
	float afInducedS[2] = {0.0f};
	CalcZInducedShift(iproj, afInducedS);
	afShift[0] += afInducedS[0];
	afShift[1] += afInducedS[1];
	CalcXInducedShift(iproj, afInducedS);
	afShift[0] += afInducedS[0];
	afShift[1] += afInducedS[1];
	// //---------------------------
	param->shiftX[iproj] += afShift[0]; 
	param->shiftY[iproj] += afShift[1];
	// clock_t part2_end_time = clock();
	// corrstack->CorrectProj(iproj, true);
	m_pCorrTomoStack->mCorrectProj(iproj);

	// int iPixels = BinSize[0] * BinSize[1];
	// size_t tybes = iPixels * sizeof(float);
	// cudaMemcpy(h_data, m_corrprojs+iproj*iPixels, tybes, cudaMemcpyDefault);
	// reproj.WriteSlice(iproj, h_data);

	// corrstack->CorrectProj(iproj, false);	 //为pnp准备,似乎原版在单独纠正图片是就没有用滤波
	// clock_t end_time = clock();
	// double part1_duration = double(part1_end_time - start_time) / CLOCKS_PER_SEC * 1000;
    // double part2_duration = double(part2_end_time - part1_end_time) / CLOCKS_PER_SEC * 1000;
    // double part3_duration = double(end_time - part2_end_time) / CLOCKS_PER_SEC * 1000;
    // double total_duration = double(end_time - start_time) / CLOCKS_PER_SEC * 1000;

    // // 输出各部分的执行时间
    // std::cout << "重投影执行时间：" << part1_duration << " 毫秒" << std::endl;
    // std::cout << "第二部分执行时间：" << part2_duration << " 毫秒" << std::endl;
    // std::cout << "第三部分执行时间：" << part3_duration << " 毫秒" << std::endl;
    // std::cout << "总执行时间：" << total_duration << " 毫秒" << std::endl;
	return fShift;   
}

void ProjAlign::MeaAlignProj(int iproj)
{
    float Tilt = angles[iproj];
    float* d_proj = m_corrprojs + iproj * BinSize[0] * BinSize[1];
	
	// size_t tBytes = PadBinSize[0] * PadBinSize[1] * sizeof(float);
	// float *PadRef, *PadImg;
    // cudaMalloc(&PadRef, tBytes);
	// cudaMalloc(&PadImg, tBytes);

	PANorm(fReproj, PadRef);
	PANorm(d_proj, PadImg);
	
	
	bool bNorm = true;
    if(!m_fft.ForwardFFT(PadRef, CmpBinSize, !bNorm))
    {
        printf("执行2D的PadRef的fft时失败 - Failed to perform 2D fft of PadRef\n");
    }
    if(!m_fft.ForwardFFT(PadImg, CmpBinSize, !bNorm))
    {
        printf("执行2D的PadImg的fft时失败 - Failed to perform 2D fft of PadImg\n");
    }

	cufftComplex* gRefCmp = (cufftComplex*)PadRef;
	cufftComplex* gImgCmp = (cufftComplex*)PadImg;
	int iPixels = PadBinSize[0] * PadBinSize[1];
    float* m_pfXcfImg = new float[iPixels];
    projgetCC(gRefCmp, gImgCmp, 500, m_pfXcfImg);    //不是五百就是四百 - It's either 500 or 400
    float Peak = PAFindPeak(m_pfXcfImg);

    delete[] m_pfXcfImg;

	// cudaFree(PadRef);
	// cudaFree(PadImg);
}

void ProjAlign::GetCentral(float* pfImg, float* gfPadImg)
{	
	// int CentSize[] = {outX, outY};
	size_t tBytes = sizeof(float) * CentSize[0];
	int iX = (BinSize[0] - CentSize[0]) / 2;  //起始点 - Starting point
	int iY = (BinSize[1] - CentSize[1]) / 2;
	int iOffset = iY * BinSize[0] + iX;
	//-------------------------------------
	for(int y=0; y<CentSize[1]; y++)
	{	float* pfSrc = pfImg + y * BinSize[0] + iOffset;
		float* gfDst = gfPadImg + y * PadBinSize[0];
		cudaMemcpy(gfDst, pfSrc, tBytes, cudaMemcpyDefault);
	}
}

void ProjAlign::PANorm(float* img, float* padimg)
{
    GetCentral(img, padimg);
	int ImgSize[] = {(PadBinSize[0] / 2 - 1) * 2, PadBinSize[1]};
	float mean = gcc.CalcMoment(padimg, ImgSize[0], ImgSize[1], PadBinSize[0], 1);
	gcc.Norm2D(padimg, PadBinSize[0], PadBinSize[1], mean, 1.0f);
	float afCent[] = {0.0f, 0.0f};
	float afMaskSize[] = {0.0f, 0.0f}; 
	afCent[0] = (PadBinSize[0] / 2 - 1) * 2 * 0.5f;
	afCent[1] = PadBinSize[1] * 0.5f;
	afMaskSize[0] = (PadBinSize[0] / 2 - 1) * 2 * m_afMaskSize[0];
	afMaskSize[1] = PadBinSize[1] * m_afMaskSize[1];
	float MaskSize_test[] = {0.0f, 0.0f};
	MaskSize_test[0] = PadBinSize[0];
	MaskSize_test[1] = PadBinSize[1];
// 	gcc.RoundEdge(padimg, PadBinSize, true, 4.0f, afCent, afMaskSize);
	gcc.RoundEdge(padimg, PadBinSize, true, 4.0f, afCent, MaskSize_test);
}

void ProjAlign::projgetCC(cufftComplex* Cmp1, cufftComplex* Cmp2, float Factor, float* m_pfXcfImg)
{
    gcc.preforCC(Cmp1, Cmp2, Factor);
    
    
    in_fft.InverseFFT(Cmp2);

	size_t tBytes = sizeof(cufftComplex) * CmpBinSize[0] * CmpBinSize[1];
	cudaMemcpy(m_pfXcfImg, Cmp2, tBytes, cudaMemcpyDefault);
}

float ProjAlign::PAFindPeak(float* m_pfXcfImg)
{	
	int m_aiXcfSize[] = {(CmpBinSize[0]-1)*2, CmpBinSize[1]};
	float fPeak = (float)-1e20;
	int aiPeak[] = {0, 0};
	int iStartX = m_aiXcfSize[0] / 20;
	int iStartY = m_aiXcfSize[1] / 20;
	if(iStartX < 3) iStartX = 3;
	if(iStartY < 3) iStartY = 3;
	int iEndX = m_aiXcfSize[0] - iStartX;
	int iEndY = m_aiXcfSize[1] - iStartY;
	//-----------------------------------
	int iPadX = (m_aiXcfSize[0] / 2 + 1) * 2;
	for(int y=iStartY; y<iEndY; y++)
	{	int i = y * iPadX;
		for(int x=iStartX; x<iEndX; x++)
		{	int j = i + x;
			if(fPeak >= m_pfXcfImg[j]) continue;
			aiPeak[0] = x;
			aiPeak[1] = y;
			fPeak = m_pfXcfImg[j];
		}
	}
	//----------------------------
	int ic = aiPeak[1] * iPadX + aiPeak[0];
        int xp = ic + 1;
        int xm = ic - 1;
        int yp = ic + iPadX;
        int ym = ic - iPadX;
	//------------------
	double a, b, c, d;
	a = (m_pfXcfImg[xp] + m_pfXcfImg[xm]) * 0.5 - m_pfXcfImg[ic];
        b = (m_pfXcfImg[xp] - m_pfXcfImg[xm]) * 0.5;
        c = (m_pfXcfImg[yp] + m_pfXcfImg[ym]) * 0.5 - m_pfXcfImg[ic];
        d = (m_pfXcfImg[yp] - m_pfXcfImg[ym]) * 0.5;
        double dCentX = -b / (2 * a + 1e-30);
        double dCentY = -d / (2 * c + 1e-30);
	//-----------------------------------
	if(fabs(dCentX) > 1) dCentX = 0;
	if(fabs(dCentY) > 1) dCentY = 0;
	float m_afPeak[2] = {0.0f};
	m_afPeak[0] = (float)(aiPeak[0] + dCentX);
	m_afPeak[1] = (float)(aiPeak[1] + dCentY);
	ishiftX = m_afPeak[0] - m_aiXcfSize[0] / 2;  //因为图像是经过中心化的，要减去图像的一半得到真实值 
	// - Because the image is centered, you need to subtract half of the image to get the real value
	ishiftY = m_afPeak[1] - m_aiXcfSize[1] / 2;
	float m_fPeak =  (float)(a * dCentX * dCentX + b * dCentX
			+ c * dCentY * dCentY + d * dCentY
			+ m_pfXcfImg[ic]);
	return m_fPeak;
}

void ProjAlign::testreproj()
{
	FitRotCenterZ();
	RemoveOffsetZ(-1.0f);
	m_afMaskSize[0] = 0.8f;
    m_afMaskSize[1] = 0.8f;
	MrcStackM reproj, coarse;
	reproj.ReadFile("/home/liuzh/tomoalign/data/reproj/proj20.mrc");
	reproj.ReadHeader();
	coarse.ReadFile("/home/liuzh/tomoalign/data/cryo-em_data/BBb/BBb_bin4_align.mrc");
	coarse.ReadHeader();

	int iPixels = coarse.X() * coarse.Y() * coarse.Z();
	float *buf1 = new float[iPixels];
	coarse.ReadBlock(0, coarse.Z(), 'z', buf1);
	cudaMemcpy(m_corrprojs, buf1, iPixels * sizeof(float), cudaMemcpyDefault);
	float *buf2 = new float[iPixels];
	reproj.ReadBlock(0, coarse.Z(), 'z', buf2);
	int ZeroTilt = GetFrameIdxFromTilt(stack->Z(), angles, 0.0f);

	float fMaxErr = (float)-1e20;
	for(int i=1; i<stack->Z(); i++)
	{	int iProj = ZeroTilt + i;
		if(iProj < stack->Z() && iProj >= 0)
		{	cudaMemcpy(fReproj, buf2+iProj*coarse.X() * coarse.Y(), sizeof(float)*coarse.X() * coarse.Y(), cudaMemcpyDefault);
			float fErr = ClacAlignProj(iProj);
			if(fErr > fMaxErr) fMaxErr = fErr;
		}
		//----------------------------------------
		iProj = ZeroTilt - i;
		if(iProj >= 0 && iProj < stack->Z())
		{	cudaMemcpy(fReproj, buf2+iProj*coarse.X() * coarse.Y(), sizeof(float)*coarse.X() * coarse.Y(), cudaMemcpyDefault);
			float fErr = ClacAlignProj(iProj);
			if(fErr > fMaxErr) fMaxErr = fErr;
		}
	}
	printf("最大误差为:%f\n", fMaxErr);
	delete[] buf1;
	delete[] buf2;
}

void ProjAlign::test()
{

}

void ProjAlign::setreproj()
{
	int pxsize = BinSize[0] * BinSize[1];
	size_t tybes = sizeof(float) * pxsize;
	h_data = new float[pxsize];
	printf("开始写入 - Start writing\n");
	// printf("测试输出值:%f\n", h_data[256]);
	reproj.InitializeHeader(BinSize[0], BinSize[1], stack->Z());
	reproj.SetSize(BinSize[0], BinSize[1], stack->Z());
	const char *name1 = "/home/liuzh/tomoalign/data/newmethod/test.mrc";
	reproj.WriteToFile(name1);
	reproj.WriteHeader();
	int ZeroTilt = GetFrameIdxFromTilt(stack->Z(), angles, 0.0f);
	cudaMemcpy(h_data, m_corrprojs+ZeroTilt*pxsize, tybes, cudaMemcpyDefault);
	reproj.WriteSlice(ZeroTilt, h_data);
	reproj.UpdateHeader();
}

void ProjAlign::closereproj()
{
	reproj.UpdateHeader();
	reproj.Close();
	delete[] h_data;
}

void ProjAlign::RotShift(float* inshift, float angle, float* outshift)
{
    float fCos = (float)cos(angle * D2R);
    float fSin = (float)sin(angle * D2R);
    float fSx = inshift[0] * fCos - inshift[1] * fSin;
    float fSy = inshift[0] * fSin + inshift[1] * fCos;
	//----------------------------------------------------
	outshift[0] = fSx;
	outshift[1] = fSy;
}

void ProjAlign::FitRotCenterZ()
{
    float afRotCent[3] = {0.0f};
	GetRotationCenter(afRotCent);
	//---------------------------------
    float afRawShift[2] = {0.0f}, afRotShift[2] = {0.0f};
    double dSin, dCos, dA, dB = 0.0;
    //------------------------------------
	//printf("RotCent的值为:%f,%f,%f\n", afRotCent[0], afRotCent[1], afRotCent[2]);
    for(int i=0; i<stack->Z(); i++)
    {       
        afRawShift[0] = param->shiftX[i];
        afRawShift[1] = param->shiftY[i];
        RotShift(afRawShift, -param->rotate[i], afRotShift);
        //-------------
		//printf("RotShift的值为:%f,%f\n", afRotShift[0], afRotShift[1]);
        dSin = sin(angles[i] * D2R);
        dCos = cos(angles[i] * D2R);
        dA += (afRotCent[0] * dCos - afRotShift[0]) * dSin;   //afRotCent[0] * dCos和afRotShift[0]的值本该相等，不相等的原因是因为z轴有偏移
		// dA += (afRotCent[0] - afRotShift[0] / dCos) * dSin;
        dB += (dSin * dSin);       //？此处不明白dA,dB是在求什么 - ? It is not clear what dA and dB are asking for here
    }
	float fOldZ = m_fZ0;
	m_fZ0 = (float)(dA / (dB + 1e-30));
	//printf("dA:%f,dB:%f\n", dA, dB);

	printf("Rot center Z: %8.2f  %8.2f  %8.2f\n\n", fOldZ, m_fZ0, m_fZ0 - fOldZ);
}

void ProjAlign::RemoveOffsetZ(float fact)
{
    if(m_fZ0 == 0) return;
    int iZeroTilt = GetFrameIdxFromTilt(stack->Z(), angles, 0.0f);
	float afInducedS[2] = {0.0f};
	for(int i=0; i<stack->Z(); i++)
	{	
		if(i == iZeroTilt) continue;
		//--------------------------
		CalcZInducedShift(i, afInducedS);
		param->shiftX[i] += (afInducedS[0] * fact);
		param->shiftY[i] += (afInducedS[1] * fact);
	}
}

void ProjAlign::FitRotCenterX()
{
	float afRotCent[3] = {0.0f};
	GetRotationCenter(afRotCent);
	//---------------------------------
	float afRawShift[2] = {0.0f}, afRotShift[2] = {0.0f};
    double dSin, dCos, dA = 0, dB = 0;
	//--------------------------------
	for(int i=0; i<stack->Z(); i++)
	{       
		afRawShift[0] = param->shiftX[i];
		afRawShift[1] = param->shiftY[i];
		RotShift(afRawShift, -param->rotate[i], afRotShift);
			//-------------
		dSin = sin(angles[i] * D2R);
		dCos = cos(angles[i] * D2R);
		dA += (afRotCent[2] * dCos * dSin + dCos * afRotShift[0]);
		dB += (dCos * dCos);
	}
	float fNewX0 = (float)(dA / (dB + 1e-30));
	m_fX0 = fNewX0 - afRotCent[0];
	printf("Rot center X: %8.2f  %8.2f  %8.2f\n\n", 
	   afRotCent[0], fNewX0, m_fX0);
}

void ProjAlign::RemoveOffsetX(float fFact)
{
	if(m_fX0 == 0) return;
	int iZeroTilt = GetFrameIdxFromTilt(stack->Z(), angles, 0.0f);
	float afShift[2] = {0.0f};
	for(int i=0; i<stack->Z(); i++)
	{	if(i == iZeroTilt) continue;
		//----------------------------
		CalcXInducedShift(i, afShift);
		param->shiftX[i] += (afShift[0] * fFact);
		param->shiftY[i] += (afShift[1] * fFact);
	}
}

void ProjAlign::CalcXInducedShift(int iFrame, float* pfShift)
{
	memset(pfShift, 0, sizeof(float) * 2);
	if(m_fX0 == 0) return;
	//--------------------
	float fTilt = angles[iFrame] * D2R;
    pfShift[0] = (float)(m_fX0 * cos(fTilt));
	pfShift[1] = 0.0f;
	RotShift(pfShift, param->rotate[iFrame], pfShift);
}

void ProjAlign::CalcZInducedShift(int iFrame, float* pfShift)
{
	memset(pfShift, 0, sizeof(float) * 2);
	if(m_fZ0 == 0) return;
	//--------------------
	float fTilt = angles[iFrame] * D2R;
	pfShift[0] = (float)(-m_fZ0 * sin(fTilt));
	RotShift(pfShift, param->rotate[iFrame], pfShift);
}	

void ProjAlign::GetRotationCenter(float* pfCenter)
{
	int iZeroTilt = GetFrameIdxFromTilt(stack->Z(), angles, 0.0f);
    float afS0[2] = {0.0f};
    //以图像中心为原点的话，位移多少就是中心偏移了多少，在correct图像的时候就是先绕着原中心旋转再位移的，所以下面就是先旋转，看看这个点被转到什么位置了
    //因为转完后图像y轴和倾斜轴平行，所以只需要对x方向的位移进行缩放
	// If the image center is taken as the origin, the displacement is the center offset. 
	// When correcting the image, it is rotated around the original center and then displaced. 
	// Therefore, the following is to rotate first and see where this point is rotated to.
	// Since after the rotation, the image y-axis is parallel to the tilt axis, only the displacement in the x-direction needs to be scaled.
    afS0[0] = param->shiftX[iZeroTilt];  
    afS0[1] = param->shiftY[iZeroTilt];
    RotShift(afS0, -param->rotate[iZeroTilt], afS0);
    pfCenter[0] = afS0[0] / cos(angles[iZeroTilt] * D2R);
    pfCenter[1] = afS0[1];
	pfCenter[2] = m_fZ0;     //z轴偏移量 - z-axis offset
}
