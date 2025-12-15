#ifndef MRCSTACK_H__
#define MRCSTACK_H__

// #include "/home/liuzh/mpi/mpi-install/include/mpi.h"
// #include "/home/xuzihe/mpi/include/mpi.h"
#include "mpi.h"
#include "mrcheader.h"
#include <iostream>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

class MrcStackM
{ // Mrc文件调用类 - Mrc file calling class
public:
    MRCheader header;
    MPI_File mpifile;
    MPI_File output;

public:
    enum Mode
    {
        MODE_BYTE = 0,
        MODE_SHORT = 1,
        MODE_FLOAT = 2
    };

public:
    MrcStackM() : mpifile(MPI_FILE_NULL), output(MPI_FILE_NULL)
    {
        header.mode = MODE_FLOAT;
    }

    ~MrcStackM() {}

    bool ReadFile(const char *filename);

    bool WriteToFile(const char *filename);

    void InitializeHeader(int width, int height, int z);

    void ReadHeader();

    bool WriteHeader();

    void UpdateHeader(bool zero=false); // only use the information from middle slice

    MRCheader &Header();

    void ReadSlice(int slcN, float *slcdata);

    void ReadoutputSlice(int slcN, float *slcdata);

    void WriteSlice(int slcN, float *slcdata);

    void ReadBlock(int start, int end, char axis,
                   float *blockdata); // not include end

    void WriteBlock(int start, int end, char axis,
                    float *blockdata); // not include end

    void WriteBlockRotX(int start, int end, char axis,
                        float *blockdata); // not include end

    void ReadAll(float *mrcdata);

    void SetSize(int nx, int ny, int nz)
    {
        header.nx = nx;
        header.ny = ny;
        header.nz = nz;
    }

    int Z() const { return header.nz; }

    void SetZ(int nz) { header.nz = nz; }

    int X() const { return header.nx; }

    void SetX(int nx) { header.nx = nx; }

    int Y() const { return header.ny; }

    void SetY(int ny) { header.ny = ny; }

    void Close();

    static void RotateX(const float *blockdata, int x, int y, int z,
                        float *rotxblock);
};

struct Point3D
{
    int x, y, z;
};

inline void AssignValue(Point3D &pt3, int _x, int _y, int _z)
{
    pt3.x = _x;
    pt3.y = _y;
    pt3.z = _z;
}

struct Point2D
{
    int x, y;
};

inline void AssignValue(Point2D &pt2, int _x, int _y)
{
    pt2.x = _x;
    pt2.y = _y;
}

struct Point3DF
{ // 三维点坐标 - 3D point coordinates
    float x, y, z;
};

inline void AssignValue(Point3DF &pt3f, int _x, int _y, int _z)
{
    pt3f.x = _x;
    pt3f.y = _y;
    pt3f.z = _z;
}

struct Point2DF
{
    float x, y;
};

inline void AssignValue(Point2DF &pt2f, int _x, int _y)
{
    pt2f.x = _x;
    pt2f.y = _y;
}

/** the order of parameters in reconstruction function should be noted
    int _x, int _y, int _z, int _length, int _width, int _height
 **/
class Volume
{ //三维体，长y 宽x 高z
  // volume will not manage the data if _data is provided
public:
    union
    { // origin
        Point3D coord;
        struct
        {
            int x, y, z;
        };
    };

    int height; //长        y
    int width;  //宽		x
    int thickness; //高		z

    float *data;
    bool external;

public:
    void Setsize(int s_width, int s_height, int s_thickness, bool cpu)
    {
        width = s_width;
        height = s_height;
        thickness = s_thickness;
        if(cpu) data = new float[width * height * thickness];
        coord.x = 0;
        coord.y = 0, coord.z = 0;
        external = false;
    }

    void SetCoord(int _x, int _y, int _z)
    {
        x = _x;
        y = _y;
        z = _z;
    }
    Volume(){}
    ~Volume()
    {
        if (!external)
        {
            delete[] data;
        }
    }
};

struct Slice
{ // slice will not manage the data if _data is provided
public:
    union
    { // origin
        Point2D coord;
        struct
        {
            int x, y;
        };
    };

    int width;  //宽		y
    int height; //长		z

    float *data;
    bool external;

public:
    Slice(int _width, int _height, float *_data)
        : width(_width), height(_height), data(_data), external(true)
    {
        coord.x = 0;
        coord.y = 0;
    }

    Slice(int _width, int _height)
        : width(_width), height(_height), external(false)
    {
        coord.x = 0;
        coord.y = 0;
        data = new float[width * height];
    }

    Slice(int _x, int _y, int _width, int _height, float *_data)
        : width(_width), height(_height), data(_data), external(true)
    {
        coord.x = _x;
        coord.y = _y;
    }

    Slice(int _x, int _y, int _width, int _height)
        : width(_width), height(_height), external(false)
    {
        coord.x = _x;
        coord.y = _y;
        data = new float[width * height];
    }

    void SetCoord(int _x, int _y)
    {
        x = _x;
        y = _y;
    }

    ~Slice()
    {
        if (!external)
        {
            delete[] data;
        }
    }
};

/*computing proj by the coordinate of a 3D pixel*/
struct Weight
{
    int x_min; // x coordinate of the proj
    int y_min; // y coordinate of the proj

    float x_min_del;
    float y_min_del; // weight of the proj
};

struct Geometry
{
    float zshift;          //表示z轴偏移  - z-axis offset
    float pitch_angle;    //表示倾斜轴偏移角 - tilt axis offset angle
    float offset;         //表示倾斜角偏移 - tilt angle offset
};

#endif
