#pragma once
#include <stdio.h>
#include <math.h>
#include <vector>
#include <cufft.h>
#include "Util.h"
#include "CorrStack.h"
#include "../mrc/mrcstack.h"

class FindOffset
{
  public:
    FindOffset();
    ~FindOffset();
    void DoIt(MrcStackM &preprojs, AlignParam* pAlignParam, std::vector<float> p_angles); 
  private:
    void FOsetup();
    float SearchOffset(int nums, float step, float initoffset);
    float CalcAverangedCC(float offset);
    float Correlate(int RefTilt, int fTilt);
  private:
    MrcStackM* stack; 
    AlignParam* param;
    std::vector<float> angles;
    CorrTomoStack* corrstack;
    float* m_corrprojs;  
    int BinSize[2];
};
