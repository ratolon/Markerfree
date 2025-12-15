#include "Transform.h"
#include "../mrc/mrcstack.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>

Transform::Transform()
{
    corrstack = 0L;
    pshiftX = 0L;
    pshiftY = 0L;
}

Transform::~Transform()
{
    if(corrstack != 0L) delete corrstack;
	corrstack = 0L;
    if(pshiftX != 0L) delete[] pshiftX;
    pshiftX = 0L;
    if(pshiftX != 0L) delete[] pshiftY;
    pshiftY = 0L;
}

void Transform::Setup(MrcStackM &preprojs, AlignParam* pAlignParam, std::vector<float> p_angles)
{
    stack = &preprojs;
    param = pAlignParam;
    angles = p_angles;

    pshiftX = new float[stack->Z()];
    pshiftY = new float[stack->Z()];
    memset(pshiftX, 0, sizeof(float) * stack->Z());
    memset(pshiftY, 0, sizeof(float) * stack->Z());
    printf("Pre-align tilt series\n");
    
    int iBinX = (int)(stack->X() / 1024.0f + 0.1f);
	int iBinY = (int)(stack->Y() / 1024.0f + 0.1f);
	int iBin = (iBinX > iBinY) ? iBinX : iBinY;
	if(iBin < 1) iBin = 1; 

    corrstack = new CorrTomoStack;
    bool randomfill = true; bool onlyshift = true; 
    bool rweight = true; bool write = true;  
    corrstack->Set0(stack, param);
    corrstack->Set1(0, iBin);
    corrstack->Set2(randomfill, onlyshift, !rweight);
    corrstack->Set3(0L, !write);
    m_corrprojs = corrstack->GetCorrectedProjs();
    corrstack->GetBinning(BinSize);
    // corrstack->stack = stack;
    // corrstack->param = param;
    // corrstack->SetSize(0.0f, iBin);
    // bool onlyshift = true; bool randomfill = true; bool write = true; 
    // corrstack->Setup(onlyshift, randomfill, !write);
    // m_corrprojs = corrstack->corrprojs;
    // BinSize[0] = corrstack->BinSize[0];
    // BinSize[1] = corrstack->BinSize[1];  
}

void Transform::DoIt()
{  
    // corrstack->CorrectProjs(); 
    corrstack->DoIt();

    float maxerr = Measure(); 
    UpdateShift();
    printf("maxerr: %8.2f\n\n", maxerr);
}

float Transform::Measure()
{
    int iPixels = BinSize[0] * BinSize[1];
    float fMaxErr = 0.0f;
    float fErr = 0.0f;
    CalcCC StretchAlign;
    StretchAlign.Setup(BinSize, 400);
    float binX = stack->X() * 1.0f / BinSize[0];
    float binY = stack->Y() * 1.0f / BinSize[1];
    for(int z=0; z<stack->Z(); z++)
    {
        int iRefProj = FindRefIndex(stack->Z(),angles, z);
        if(iRefProj == z) continue;

        float RefTilt = angles[iRefProj];
        float fTilt = angles[z];
        float fTiltAxis = param->rotate[z];
        float* d_reproj = m_corrprojs + iRefProj * iPixels;
        float* d_proj = m_corrprojs + z * iPixels;

        StretchAlign.DoIt(d_reproj, d_proj, RefTilt, fTilt, fTiltAxis);
        StretchAlign.getshift(pshiftX[z], pshiftY[z], binX, binY);
        printf("Image %4d shift: %8.2f, %8.2f px\n", z+1, pshiftX[z], pshiftY[z]);
        fErr = (float)sqrt(pshiftX[z] * pshiftX[z] + pshiftY[z] * pshiftY[z]);
        if(fErr > fMaxErr) fMaxErr = fErr;
    }
    return fMaxErr;
}

void Transform::UpdateShift(void)
{
    int ZeroTilt = GetFrameIdxFromTilt(stack->Z(), angles, 0.0f);
    for(int i=0; i<stack->Z(); i++){
        pshiftX[i] -= pshiftX[ZeroTilt];
        pshiftY[i] -= pshiftY[ZeroTilt];
    } 
    // printf("zerotilt:%d,zerotrans:%f,%f\n", ZeroTilt, pshiftX[ZeroTilt], pshiftY[ZeroTilt]);
    float fSumShift[2] = {0.0f};
    for(int i=ZeroTilt-1; i>=0; i--)
    {
        // printf("%d张图累加前的SumShift位移:%f,%f\n", i+1, fSumShift[0], fSumShift[1]);
        // printf("%d张图累加前的shift位移:%f,%f\n", i+1, pshiftX[i], pshiftY[i]);
        fSumShift[0] += pshiftX[i];
        fSumShift[1] += pshiftY[i];
        // printf("%d张图UnStretch前的SumShift位移:%f,%f\n", i+1, fSumShift[0], fSumShift[1]);
        
        UnStretch(i+1, i, fSumShift);
        // printf("%d张图UnStretch后的SumShift位移:%f,%f\n\n", i+1, fSumShift[0], fSumShift[1]);
        param->shiftX[i] += fSumShift[0];
        param->shiftY[i] += fSumShift[1];
        
    }
    memset(fSumShift, 0, sizeof(fSumShift));
    for(int i=ZeroTilt+1; i<stack->Z(); i++)
    {
        fSumShift[0] += pshiftX[i];
        fSumShift[1] += pshiftY[i];
        UnStretch(i-1, i, fSumShift);
        param->shiftX[i] += fSumShift[0];
        param->shiftY[i] += fSumShift[1];
    }
}

void Transform::UnStretch(int ref, int f, float* SumShift)
{
    float fRefTilt = angles[ref];
    float fTilt = angles[f];
    float fTiltAxis = param->rotate[f];
    double dStretch = cos(fRefTilt * D2R) / cos(fTilt * D2R);
    //-------------------------------------------------------------
    double d2T = 2 * D2R * fTiltAxis;
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

    float buf = SumShift[0] * Matrix[0] + SumShift[1] * Matrix[1];
    SumShift[1] = SumShift[0] * Matrix[1] + SumShift[1] * Matrix[2];
    SumShift[0] = buf;
}
