#include "../method/Util.h"
#include "../method/CorrStack.h"
#include "../method/Transform.h"
#include "../method/Rotate.h"
#include "../method/FindOffset.h"
#include "../mrc/mrcstack.h"
#include "opts.h"
#include "Process.h"
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <random>
#include <unistd.h>
#include <limits.h>
#include <libgen.h> // Provides dirname() function

Process::Process()
{
    param = new AlignParam();
}
Process::~Process()
{    
    if (param->rotate != nullptr) delete[] param->rotate;
    if (param->shiftX != nullptr) delete[] param->shiftX;
    if (param->shiftY != nullptr) delete[] param->shiftY;
    delete param;
}

void Process::DoIt(options &opt, SysInfo &info)
{
    struct timeval start_time, end_time;
    gettimeofday(&start_time, nullptr);

    // ReadStack(opt, info);

    int status = ReadStack(opt, info);

    // 🔥 如果读取失败，必须立即所有 rank 全部退出
    // If reading fails, all ranks must exit immediately
    if (status != 0)
    {
        if (info.id == 0)
            printf("[Abort] DoIt() stopped due to input file loading failure.\n");

        return; // ❗❗立刻退出，不再跑下面所有步骤
        // ❗❗ Exit immediately, do not run all the following steps
    }

    Geometry geo;
    geo.offset = opt.offset;  //表示倾斜角偏移 - Offset of tilt axis
    geo.pitch_angle = opt.pitch_angle;   //表示倾斜轴偏移角 - Pitch angle of tilt axis
    geo.zshift = opt.zshift;  //表示z轴偏移 - Z shift
    ReadAngles(p_angles, opt.angle);

    bool angle_ok = ReadAngles(p_angles, opt.angle);
    if (!angle_ok)
    {
        if (info.id == 0)
            printf("[Error] Cannot open angle file: %s\n", opt.angle);

        printf("[Abort] DoIt() stopped due to angle file loading failure.\n");
        return;
    }

    mPreprocess();  //图像预处理 - Preprocess images

    if(opt.AlignZ != 0)
    {
        SetParam();   //初始化参数 - Initialize alignment parameters

        mCoarseAlign(geo);  //图像粗对齐 - Coarse alignment

        mProjAlign(geo, opt);  //图像投影匹配对齐 - Projection matching alignment
    }

    CorrectStack(opt);  //图像矫正和保存 - Correct and save images

    gettimeofday(&end_time, nullptr);
    long seconds = end_time.tv_sec - start_time.tv_sec;
    long microseconds = end_time.tv_usec - start_time.tv_usec;
    double elapsed_time = seconds + microseconds / 1e6;
    if(opt.AlignZ != 0) CollectParam(opt, elapsed_time);  //收集对齐相关参数和时间 - Collect alignment parameters and time
    std::cout << "Elapsed time: " << elapsed_time << " seconds" << std::endl;

    preprojs.Close();
    // MPI_Barrier(MPI_COMM_WORLD);
}

// int Process::ReadStack(options &opt, SysInfo &info)
// {
//     //MrcStackM projs;
//     if (!projs.ReadFile(opt.input))  //打开mrc文件，将mpi句柄赋给projs的参数mpifile
//     {
//         printf("File %s cannot access.\n", opt.input);

//         return -1;
//     }

//     if (info.id == 0)
//     {
//         projs.ReadHeader();
//     }
//     MPI_Bcast(&(projs.header), sizeof(MRCheader), MPI_CHAR, 0, MPI_COMM_WORLD);

//     preprojs.InitializeHeader(projs.X(), projs.Y(), projs.Z());
//     preprojs.SetSize(projs.X(), projs.Y(), projs.Z()); 
//     // std::string bufFilePath = extractParentFolder(opt.output) + "/buf.mrc";

//     char absolutePath[PATH_MAX];
//     realpath(opt.output, absolutePath);      // opt.output 已经是 C 风格字符串
//     std::string dir = dirname(absolutePath); // 需要 #include <libgen.h>
//     std::string bufFilePath = std::string(dir) + "/buf.mrc";

//     preprojs.WriteToFile(bufFilePath.c_str());

//     if (info.id == 0)
//     {
//         preprojs.WriteHeader();
//     }

//     return 0;
// }

int Process::ReadStack(options &opt, SysInfo &info)
{
    int status = 0; // 0 表示正常，负值表示错误 - 0 means normal, negative means error

    // Rank 0 负责读文件 - Rank 0 is responsible for reading the file
    if (info.id == 0)
    {
        if (!projs.ReadFile(opt.input))
        {
            printf("[Error] Cannot open input file: %s\n", opt.input);
            status = -1;
        }
    }

    // 广播文件读取状态给所有 rank - Broadcast file reading status to all ranks
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (status != 0)
    {
        if (info.id == 0)
            printf("[Abort] Input stack loading failed. Please check the file path.\n");
        return status;
    }

    if (info.id == 0)
        projs.ReadHeader();

    MPI_Bcast(&(projs.header), sizeof(MRCheader), MPI_CHAR, 0, MPI_COMM_WORLD);

    preprojs.InitializeHeader(projs.X(), projs.Y(), projs.Z());
    preprojs.SetSize(projs.X(), projs.Y(), projs.Z());

    // 构造 buf.mrc 的绝对路径 - Construct the absolute path of buf.mrc
    char absolutePath[PATH_MAX];
    realpath(opt.output, absolutePath);
    std::string dir = dirname(absolutePath);
    std::string bufFilePath = dir + "/buf.mrc";

    preprojs.WriteToFile(bufFilePath.c_str());

    if (info.id == 0)
        preprojs.WriteHeader();

    return 0;
}

std::string Process::extractParentFolder(const char* filename)
{
    std::string pathString(filename);
    size_t found = pathString.find_last_of("/\\"); // 查找最后一个路径分隔符 - Find the last path separator
    if (found != std::string::npos) {
        // 截取上一级文件夹名称 - Extract the parent folder name
        return pathString.substr(0, found);
    }
    return "";
}

bool Process::ReadAngles(std::vector<float> &angles, const char *name)
{
  std::ifstream in(name);
  if (!in.good())
  {
    return false;
  }

  while (in.good())
  {
    float val;
    in >> val;
    if (in.fail())
    {
      break;
    }
    angles.push_back(val);
  }
  in.close();
  return true;
}

void Process::SetParam()
{
    int nz = preprojs.Z();
    param->shiftX = new float[nz];
    param->shiftY = new float[nz];
    param->rotate = new float[nz];

    // 初始化数组为 0 - Initialize arrays to 0
    for (int i = 0; i < nz; i++) {
        param->shiftX[i] = 0.0f;
        param->shiftY[i] = 0.0f;
        param->rotate[i] = 0.0f;
    } 
}

void Process::mPreprocess()
{
    PreProcess preprocess;
    preprocess.rawstack = &projs;
    preprocess.stack = &preprojs;
    preprocess.angles = p_angles;
    preprocess.SetPositive();
    preprocess.MassNormalization();
    projs.Close();
    if(p_angles[0]>p_angles[1])
    {
        reverse(p_angles.begin(), p_angles.end());
    }
}

void Process::mCoarseAlign(Geometry &geo)   //这一过程只修改了参数param - This process only modifies the parameters in param
{  
    if(geo.pitch_angle == 0)
    {
        Transform transform;
        transform.Setup(preprojs, param, p_angles);
        CalcTIltAxis rotate;
        for(int i=1; i<=3; i++)
        {
            transform.DoIt();
            rotate.DoIt(preprojs, param, p_angles, 180.0f / i, 100);
        }   
    }
    else
    {
        for (int i = 0; i < preprojs.Z(); i++) {
            param->rotate[i] = geo.pitch_angle;
        }
    }
    if(geo.offset == 0)
    {   
        FindOffset findoffset;
        findoffset.DoIt(preprojs, param, p_angles);
    }
    else
    {
        param->angleOffset = geo.offset;
    }
    for(int z=0; z<preprojs.Z(); z++)
    {
        p_angles[z] += param->angleOffset;
    }
}

void Process::mProjAlign(Geometry &geo, options &opt)
{
    ResetShift();
    ProjAlign projalign;
    CalcTIltAxis rotate;
    //projalign.outfile=opt.output;
    projalign.SetNProj(opt.nProj);
    projalign.Setup(preprojs, param, p_angles, geo.zshift, opt.AlignZ);   
    // projalign.test(); 
    float fRange = 20.0f;
    int iIters = 4;

    ProjAlignOnce(projalign);
    for(int i=1; i<=iIters; i++) 
    {	rotate.DoIt(preprojs, param, p_angles, fRange / i, 100);
        if(i == 1) ProjAlignOnce(projalign);
    }
    ProjAlignOnce(projalign);
}

void Process::ProjAlignOnce(ProjAlign &projalign)
{
    projalign.m_afMaskSize[0] = 0.7f;
    projalign.m_afMaskSize[1] = 0.7f;
    float fLastErr = projalign.DoIt();
    printf("fLastErr为: %8.2f\n", fLastErr);

    AlignParam* LastParam = NewCopyParam();
    int iIterations = 10;
    projalign.m_afMaskSize[0] = 0.55f;
    projalign.m_afMaskSize[1] = 0.55f;
    for(int i=1; i<iIterations; i++)
	{	float fErr = projalign.DoIt();
        printf("fErr为: %8.2f\n", fErr);
		// if(fErr < 2.0f) break;
		//--------------------
		if(fErr <= fLastErr) 
        {
            if(fErr < 2.0f) break;
            fLastErr = fErr;
            CopyParam(LastParam, param);
        }
	}
    delete[] LastParam->shiftX;
    delete[] LastParam->shiftY;
    delete[] LastParam->rotate;
    delete LastParam;
}

void Process::ResetShift()
{
    for (int i = 0; i < preprojs.Z(); i++) {
        param->shiftX[i] = 0.0f;
        param->shiftY[i] = 0.0f;
    }
}

void Process::CorrectStack(options &opt)
{
    CorrTomoStack corr;
    MrcStackM binprojs;

    int binsize[2] = {0};
    float fTiltAxis = param->rotate[preprojs.Z() / 2];
    bool randomfill = true; bool onlyshift = true; 
    bool rweight = true; bool write = true;  

    corr.Set0(&preprojs, param);
    corr.Set1(fTiltAxis, opt.OutBin);
    corr.Set2(randomfill, !onlyshift, !rweight);
    corr.Set3(&binprojs, write);

    corr.GetBinning(binsize);
    binprojs.InitializeHeader(binsize[0], binsize[1], preprojs.Z());
    binprojs.SetSize(binsize[0], binsize[1], preprojs.Z()); 
    binprojs.WriteToFile(opt.output);
    binprojs.WriteHeader();
    corr.DoIt();

    binprojs.UpdateHeader();

    char absolutePath[PATH_MAX];

    if (realpath(opt.output, absolutePath))
    {
        std::string bufFilePath = std::string(dirname(absolutePath)) + "/buf.mrc";
        if (std::remove(bufFilePath.c_str()) == 0)
        {
            std::cout << "File " << bufFilePath << " successfully deleted." << std::endl;
        }
        else
        {
            std::cerr << "Error deleting file " << bufFilePath << std::endl;
        }
    }
    else
    {
        std::cerr << "Error resolving absolute path for " << opt.output << std::endl;
    }
    binprojs.Close();
}

AlignParam* Process::NewCopyParam()
{
    AlignParam* pAlignParam = new AlignParam;
    int nz = preprojs.Z();
    pAlignParam->shiftX = new float[nz];
    pAlignParam->shiftY = new float[nz];
    pAlignParam->rotate = new float[nz];

    for (int i = 0; i < nz; i++) {
        pAlignParam->shiftX[i] = 0.0f;
        pAlignParam->shiftY[i] = 0.0f;
        pAlignParam->rotate[i] = 0.0f;
    } 
    CopyParam(pAlignParam, param);
    return pAlignParam;
}

void Process::CopyParam(AlignParam* dAlignParam, AlignParam* sAlignParam)
{
    int iBytes = preprojs.Z() * sizeof(float);
    memcpy(dAlignParam->rotate, sAlignParam->rotate, iBytes);
	memcpy(dAlignParam->shiftX, sAlignParam->shiftX, iBytes);
	memcpy(dAlignParam->shiftY, sAlignParam->shiftY, iBytes);
}

void Process::CollectParam(options &opt, double time)
{
    SaveParam saveparam;
    saveparam.output = &preprojs;
    saveparam.param = param;
    saveparam.outfile = opt.output;
    saveparam.iBin = opt.OutBin;
    saveparam.time = (float)time;
    std::vector<float> angles;
    ReadAngles(angles, opt.angle);
    bool inv = false;
    if(angles[0]>angles[1])
    {
        reverse(angles.begin(), angles.end());
        inv = true;
    }
    saveparam.angles = angles;
    saveparam.GetParam(1, opt.Savemode, inv);  //4表示一开始输入的图像缩小了四倍，在收集位移数据的时候要乘4
    // 4 means the input images were downsampled by a factor of 4 at the beginning, so multiply the shift data by 4 when collecting
}

void TranslateAngleToCoefficients(const std::vector<float> &angles,
								  const std::vector<float> &xangles,
								  std::vector<Coeff> &coeffs)
{
	coeffs.resize(angles.size());
	for (int i = 0; i < angles.size(); i++)
	{
		memset(coeffs[i].p, 0, sizeof(double) * 20);
		float beta = D2R * angles[i];
		float alpha = D2R * xangles[i];

		coeffs[i].a[0] = 0;						  //
		coeffs[i].a[1] = cos(beta);				  // x
		coeffs[i].a[2] = sin(alpha) * sin(beta);  // y
		coeffs[i].a[3] = -cos(alpha) * sin(beta); // z
		coeffs[i].b[0] = 0;						  //
		coeffs[i].b[1] = 0;						  // x
		coeffs[i].b[2] = cos(alpha);			  // y
		coeffs[i].b[3] = sin(alpha);			  // z
	}
}
