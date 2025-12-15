#include "Util.h"
#include "../mrc/mrcstack.h"
#include <stdio.h>
#include <cufft.h>
#include <math.h>
#include <assert.h>

int GetFrameIdxFromTilt(int nz, std::vector<float> p_angles, float fTilt)
{
	int iFrameIdx = 0;
	float fMin = (float)fabs(fTilt - p_angles[0]);
	for(int i=1; i<nz; i++)
	{	float fDiff = (float)fabs(fTilt - p_angles[i]);
		if(fDiff >= fMin) continue;
		fMin = fDiff;
		iFrameIdx = i;
	}
    //printf("最接近%f的角度为%f\n", fTilt, angles[iFrameIdx]);
	return iFrameIdx;
}

int FindRefIndex(int nz, std::vector<float> p_angles, int z) //找到相邻两张图像中倾角绝对值比本身小的那个,也就是更靠近零倾斜的那个 - 
// Find the one with a smaller absolute tilt angle among the two adjacent images, that is, the one closer to zero tilt
{
	int iProj0 = z - 1;    //找到相邻图像中更接近零倾斜的那个图像，作为相邻图像 - Find the image closer to zero tilt among adjacent images as adjacent images
    int iProj2 = z + 1;
    if(iProj0 < 0) iProj0 = z;
    if(iProj2 >= nz) iProj2 = z;
    double dTilt0 = fabs(p_angles[iProj0]);
    double dTilt = fabs(p_angles[z]);
    double dTilt2 = fabs(p_angles[iProj2]);
    if(dTilt0 < dTilt) return iProj0;   
    else if(dTilt2 < dTilt) return iProj2;
    else return z;
}
