#pragma once
#include <stdio.h>
#include <math.h>
#include <vector>
#include <cufft.h>
#include "Util.h"
#include "CorrStack.h"
#include "../mrc/mrcstack.h"

class Transform
{
  public:
    Transform();
    ~Transform();
    void Setup(MrcStackM &preprojs, AlignParam* pAlignParam, std::vector<float> p_angles);
    void DoIt();    
  private:
    float Measure();
    void UpdateShift();
    void ResetShift();
    void UnStretch(int i, int j, float* SumShift);
  private:
    MrcStackM* stack; 
    AlignParam* param;
    std::vector<float> angles;
    CorrTomoStack* corrstack;
    float* pshiftX;  //用于存储每一个循环得到的位移参数，随后加到总的位移参数上 -
    //  Used to store the displacement parameters obtained in each loop, which are then added to the total displacement parameters
    float* pshiftY;
    int BinSize[2];
    float* m_corrprojs;  
};