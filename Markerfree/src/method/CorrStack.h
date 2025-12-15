#pragma once
#include <stdio.h>
#include <math.h>
#include <vector>
#include <cufft.h>
#include "Cufft1&2.h"
#include "../mrc/mrcstack.h"
#include "Util.h"

class CorrUtil
{
public:
	static void CalcAlignedSize
	(int* piRawSize, float fTiltAxis, int* piAlnSize);
	static void CalcBinnedSize
	(int* piRawSize, float fBinning, int* piBinnedSize);
	static void Unpad
	(float* pfPad, int* piPadSize, float* pfImg);
};

class GCorrPatchShift
{
public:
	GCorrPatchShift();
	~GCorrPatchShift();
	void SetSizes
	( int* piInSize,
	  bool bInPadded,
	  int* piOutSize,
	  bool bOutPadded
	//   int iNumPatches  //这个参数与局部对齐有关 - This parameter is related to local alignment
	);
	void DoIt
	( float* gfInImg,
	  float* pfGlobalShift,
	  float fRotAngle,
	//   float* gfLocalAlnParams,
	  bool bRandomFill,
	  float* gfOutImg
	);
private:
	// float m_fD2R;
	int m_iInImgX;
	int m_iOutImgX;
	int m_iOutImgY;
};

class BinImage
{
public:
	BinImage();
	~BinImage();
	static void GetBinSize
	(int* piInSize, bool bInPadded, int iBinning,
	 int* piOutSize, bool bOutPadded
	);
	void SetupBinnings
	(int* piInSize,  bool bInPadded,
	 int* piBinning, bool bOutPadded
	);
	void SetupBinning
	(int* piInSize, bool bInPadded,
	 int iBinning, bool bOutPadded
	);
	void SetupSizes
	(int* piInSize, bool bInPadded,
	 int* piOutSize, bool bOutPadded
	);
	void DoIt
	(float* gfInImg,  // input image
	 float* gfOutImg
	);
	int m_aiOutSize[2];
	int m_iOutImgX;
	int m_aiBinning[2];
};

class RWeight
{
  public:
    RWeight();
    ~RWeight();
    void Setup(int iPadProjX, int iNumProjs);
    void DoIt(float* gfPadSinogram);
  private:
    cufft1D* pGForward;
    cufft1D* pGInverse;
    cufft2D forward;
    cufft2D inverse;
    int CmpSize[2];
};

class CorrTomoStack
{
public:
    CorrTomoStack();
    ~CorrTomoStack();
    void Set0(MrcStackM* stack, AlignParam* param);
    void Set1(float fTiltAxis, float fOutBin);
    void Set2(bool bRandFill, bool bShiftOnly, bool bRweight);
    void Set3(MrcStackM* stack, bool write);
    void DoIt();
	void mCorrectProj(int iProj);
	void GetBinning(int* pfBinning);
	float* GetCorrectedProjs();
    void Clean();

private:
    // void mCorrectProj(int iProj);
    MrcStackM* mstack;
	MrcStackM* m_WriteFile;
    AlignParam* mparam;
    GCorrPatchShift m_aGCorrPatchShift;
	BinImage m_aGBinImg2D;
    RWeight* m_pRrweight;
    int m_aiStkSize[3];
    int m_aiAlnSize[3];
	int m_aiBinnedSize[3];
	float m_afBinning[2];

    float* m_gfRawProj; //一张初始图像，内存分配无pad - An initial image, memory allocated without pad
	float* m_gfCorrProj; //一张纠正的图像，内存分配有pad - A corrected image, memory allocated with pad
	float* m_gfBinProj; //一张纠正后经过尺寸缩小处理的图像，内存分配有pad - A corrected and downsampled image, memory allocated with pad
	float* m_gfCorrectProjs;  //最后的纠正图像结果，全部图像，内存分类无pad - The final corrected image result, all images, memory allocated without pad

	float m_fOutBin;
    bool m_onlyshift;
    bool m_bRandomFill;
    bool m_bWrite = false;

	float* h_proj;
	float* h_newproj;
};
