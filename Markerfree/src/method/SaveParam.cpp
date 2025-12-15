#include "Util.h"
#include "../mrc/mrcstack.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

SaveParam::SaveParam()
{
}

SaveParam::~SaveParam()
{
}

void SaveParam::SaveXf(bool inv)
{
	if(param == 0L) return;
	FILE* pFile = (FILE*)m_pvFile;
	float afShift[] = {0.0f, 0.0f};
	const float width = output->X();  // 替换为实际图像宽度 - Replace with actual image width
	const float height = output->X(); // 替换为实际图像高度 - Replace with actual image height
	const float cx = width / 2.0f;
	const float cy = height / 2.0f;

	const int z_size = output->Z();
	const int start = inv ? z_size - 1 : 0;
	const int end = inv ? -1 : z_size;
	const int step = inv ? -1 : 1;

	for (int i = start; i != end; i += step) {
		float fTilt = angles[i];
		float fTiltAxis = param->rotate[i];
		float par[6];
		afShift[0] = param->shiftX[i] * mbin;
		afShift[1] = param->shiftY[i] * mbin;

		float theta_rad = fTiltAxis * M_PI / 180.0f; 
		float cos_theta = cos(theta_rad);
		float sin_theta = sin(theta_rad);

		float T1 = -afShift[0] - cx + cos_theta * cx - sin_theta * cy;
		float T2 = -afShift[1] - cy + sin_theta * cx + cos_theta * cy;

		par[0] = cos_theta;
		par[1] = sin_theta;
		par[2] = -sin_theta;
		par[3] = cos_theta;
		par[4] = T1 * cos_theta + T2 * sin_theta - cx + cos_theta * cx + sin_theta * cy;
		par[5] = -T1 * sin_theta + T2 * cos_theta - cy - sin_theta * cx + cos_theta * cy;
		
		fprintf(pFile, "%9.3f  %9.3f  %9.3f  %9.3f  %9.3f  %9.3f \n",  
			par[0], par[1], par[2], par[3], par[4], par[5]);
	}


	// if(inv)
	// {
	// 	for(int i = output->Z() - 1; i >= 0; i--) 
	// 	{
	// 		float fTilt = angles[i];
	// 		float fTiltAxis = param->rotate[i];
	// 		float par[6];
	// 		afShift[0] = param->shiftX[i] * mbin;
	// 		afShift[1] = param->shiftY[i] * mbin;

	// 		float theta_rad = fTiltAxis * M_PI / 180.0f; 
		
	// 		// 计算三角函数值
	// 		float cos_theta = cos(theta_rad);
	// 		float sin_theta = sin(theta_rad);

	// 		// 计算平移参数
	// 		float T1 = -afShift[0] - cx + cos_theta * cx - sin_theta * cy;
	// 		float T2 = -afShift[1] - cy + sin_theta * cx + cos_theta * cy;

	// 		// 计算最终XF参数
	// 		par[0] = cos_theta;
	// 		par[1] = sin_theta;
	// 		par[2] = -sin_theta;
	// 		par[3] = cos_theta;
	// 		par[4] = T1 * cos_theta + T2 * sin_theta - cx + cos_theta * cx + sin_theta * cy;
	// 		par[5] = -T1 * sin_theta + T2 * cos_theta - cy - sin_theta * cx + cos_theta * cy;
			
	// 		fprintf(pFile, "%9.3f  %9.3f  %9.3f  %9.3f  %9.3f  %9.3f \n",  
	// 		par[0], par[1], par[2], par[3], par[4], par[5]);
	// 	}
	// }
	// else{
	// 	for(int i=0; i<output->Z(); i++)
	// {
	// 		float fTilt = angles[i];
	// 		float fTiltAxis = param->rotate[i];
	// 		float par[6];
	// 		afShift[0] = param->shiftX[i] * mbin;
	// 		afShift[1] = param->shiftY[i] * mbin;

	// 		float theta_rad = fTiltAxis * M_PI / 180.0f; 
		
	// 		// 计算三角函数值
	// 		float cos_theta = cos(theta_rad);
	// 		float sin_theta = sin(theta_rad);

	// 		// 计算平移参数
	// 		float T1 = -afShift[0] - cx + cos_theta * cx - sin_theta * cy;
	// 		float T2 = -afShift[1] - cy + sin_theta * cx + cos_theta * cy;

	// 		// 计算最终XF参数
	// 		par[0] = cos_theta;
	// 		par[1] = sin_theta;
	// 		par[2] = -sin_theta;
	// 		par[3] = cos_theta;
	// 		par[4] = T1 * cos_theta + T2 * sin_theta - cx + cos_theta * cx + sin_theta * cy;
	// 		par[5] = -T1 * sin_theta + T2 * cos_theta - cy - sin_theta * cx + cos_theta * cy;
			
	// 		fprintf(pFile, "%9.3f  %9.3f  %9.3f  %9.3f  %9.3f  %9.3f \n",  
	// 		par[0], par[1], par[2], par[3], par[4], par[5]);
	// 	}
	// }

}

void SaveParam::GetParam(int bin, int mode, bool inv)
{
	mbin = bin;
    char filename[256] = {'\0'};
    strcpy(filename, outfile);
    char* pcOutSlash = strrchr(filename, '/');
//	if(pcOutSlash == 0L) printf("Error output path");


	if(mode == 0){
		char* pcMrc = strstr(filename, ".mrc");
		if(pcMrc == 0L) 
			strcat(filename, ".txt");
		else 
			strcpy(pcMrc, ".txt");
		FILE* pFile = fopen(filename, "wt");
		if(pFile == 0L)
		{	printf("Unable to open %s.\n", filename);
			printf("Alignment data will not be saved\n\n");
			return;
		}
		m_pvFile = pFile;
		SaveHeader();
		SaveAllParam();
		SaveTime();
		CloseFile();
	}
	else if((mode == 1)){
		char* pcMrc = strstr(filename, ".mrc");
		if(pcMrc == 0L) 
			strcat(filename, ".xf");
		else 
			strcpy(pcMrc, ".xf");
		FILE* pFile = fopen(filename, "wt");
		if(pFile == 0L)
		{	printf("Unable to open %s.\n", filename);
			printf("Alignment data will not be saved\n\n");
			return;
		}
		m_pvFile = pFile;
		SaveXf(inv);
		CloseFile();
	}

    // char* pcMrc = strstr(filename, ".mrc");
    // if(pcMrc == 0L) 
    //     strcat(filename, ".txt");
    // else 
    //     strcpy(pcMrc, ".txt");
    // FILE* pFile = fopen(filename, "wt");
    // if(pFile == 0L)
	// {	printf("Unable to open %s.\n", filename);
	// 	printf("Alignment data will not be saved\n\n");
	// 	return;
	// }
    // m_pvFile = pFile;
    // // SaveHeader();
    // // SaveAllParam();
    // // SaveTime();
	// SaveXf();
    // CloseFile();
}

void SaveParam::SaveHeader()
{
	FILE* pFile = (FILE*)m_pvFile;
	fprintf(pFile, "# Markerfree Alignment\n");
	fprintf(pFile, "# RawSize = %d %d %d\n", output->X(),output->Y(),output->Z());
	//---------------------------------------------------
}

// void SaveParam::SaveAllParam()
// {
// 	if(1){
// 		if(param == 0L) return;
// 		FILE* pFile = (FILE*)m_pvFile;
// 		fprintf( pFile, "# SEC       ROT          "
// 		"TX        TY         TILT\n");
// 		//--------------------------------------------------------------------
// 		float afShift[] = {0.0f, 0.0f};
// 		for(int i=0; i<output->Z(); i++)
// 		{
// 			float fTilt = angles[i];
// 			float fTiltAxis = param->rotate[i];
// 			afShift[0] = param->shiftX[i] * mbin;
// 			afShift[1] = param->shiftY[i] * mbin;
// 			fprintf( pFile, "%5d   %9.4f  %9.3f  %9.3f   "
// 			"%8.2f\n", i, fTiltAxis, 
// 			afShift[0], afShift[1], fTilt);
// 		}
// 		fprintf(pFile, "angleoffset: %f\n", param->angleOffset);
// 		fprintf(pFile, "Output Bin: %d\n", iBin);
// 	}
// 	else{
// 		if(param == 0L) return;
// 		FILE* pFile = (FILE*)m_pvFile;
// 		float afShift[] = {0.0f, 0.0f};
// 		for(int i=0; i<output->Z(); i++)
// 		{
// 			float par[6];
// 			float fTilt = angles[i];
// 			float fTiltAxis = param->rotate[i];
// 			afShift[0] = param->shiftX[i] * mbin;
// 			afShift[1] = param->shiftY[i] * mbin;

// 			txt2xf(fTiltAxis, afShift[0], afShift[1], par);

// 			// fprintf( pFile, "%5d   %9.4f  %9.3f  %9.3f   "
// 			// "%8.2f\n", i, fTiltAxis, 
// 			// afShift[0], afShift[1], fTilt);
// 		}
// 	}
// }

void SaveParam::SaveAllParam()
{
	if(param == 0L) return;
	FILE* pFile = (FILE*)m_pvFile;
	fprintf( pFile, "# SEC       ROT          "
	"TX        TY         TILT\n");
	//--------------------------------------------------------------------
	float afShift[] = {0.0f, 0.0f};
	for(int i=0; i<output->Z(); i++)
	{
		float fTilt = angles[i];
		float fTiltAxis = param->rotate[i];
		afShift[0] = param->shiftX[i] * mbin;
		afShift[1] = param->shiftY[i] * mbin;
		fprintf( pFile, "%5d   %9.4f  %9.3f  %9.3f   "
		"%8.2f\n", i, fTiltAxis, 
		afShift[0], afShift[1], fTilt);
	}
	fprintf(pFile, "angleoffset: %f\n", param->angleOffset);
	fprintf(pFile, "Output Bin: %d\n", iBin);
}

void SaveParam::SaveTime()
{
    FILE* pFile = (FILE*)m_pvFile;
    fprintf(pFile, "Elapsed time: %f seconds\n", time);
}

void SaveParam::CloseFile()
{
	if(m_pvFile == 0L) return;
	fclose((FILE*)m_pvFile);
	m_pvFile = 0L;
}
