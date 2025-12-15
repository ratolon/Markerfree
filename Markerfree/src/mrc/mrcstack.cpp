#include "mrcstack.h"
#include <cfloat>
#include <limits>

bool MrcStackM::ReadFile(const char *filename)
{
    int rc = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY,
                           MPI_INFO_NULL, &mpifile);
    return !rc;
}

bool MrcStackM::WriteToFile(const char *filename)  //函数中的mpifile原来是output
{
    int rc =
        MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_RDWR,
                      MPI_INFO_NULL, &output);

    size_t psize;
    switch (header.mode)
    {
    case MODE_BYTE:
        psize = sizeof(char);
        break;
    case MODE_SHORT:
        psize = sizeof(short);
        break;
    case MODE_FLOAT:
        psize = sizeof(float);
        break;

    default:
        printf("File type unknown!\n");
        return false;
    }

    MPI_File_set_size(output, sizeof(MrcHeader) + header.next +
                                  psize * header.nx * header.ny * header.nz);
    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDWR, MPI_INFO_NULL,
                  &output);
    return !rc;
}

void MrcStackM::ReadHeader()
{
    MPI_File_seek(mpifile, 0, MPI_SEEK_SET);
    MPI_File_read(mpifile, &header, sizeof(MRCheader), MPI_CHAR,
                  MPI_STATUS_IGNORE);
}
MRCheader &MrcStackM::Header() { return header; }

void MrcStackM::ReadSlice(int slcN, float *slcdata)
{
    MPI_Offset glboffset = sizeof(MRCheader) + header.next;
    int psize;

    switch (header.mode)
    {
    case MODE_BYTE:
        psize = sizeof(char);

        break;

    case MODE_SHORT:
        psize = sizeof(short);

        break;

    case MODE_FLOAT:
        psize = sizeof(float);

        break;

    default:
        printf("File type unknown!\n");
        return;
    }

    char *buf;
    size_t bufsize;
    char *cur;

    size_t slcsize = header.nx * header.ny * psize;


    bufsize = header.nx * header.ny;
    buf = new char[bufsize * psize];

    MPI_File_seek(mpifile, glboffset + slcN * slcsize, MPI_SEEK_SET);
    MPI_File_read(mpifile, buf, slcsize, MPI_CHAR, MPI_STATUS_IGNORE);
    
    switch (header.mode)
    {
    case MODE_BYTE:
        for (int i = bufsize; i--;)
        {
            slcdata[i] = ((unsigned char *)buf)[i];
        }

        break;

    case MODE_SHORT:
        for (int i = bufsize; i--;)
        {
            slcdata[i] = ((short *)buf)[i];
        }

        break;

    case MODE_FLOAT:
        memcpy(slcdata, buf, sizeof(float) * bufsize);

        break;

    default:
        printf("File type unknown!\n");
    }

    delete[] buf;
}

void MrcStackM::ReadoutputSlice(int slcN, float *slcdata)
{
    int psize = sizeof(float);
    MPI_Offset glboffset = sizeof(MRCheader) + header.next;
    char *buf;
    size_t bufsize;
    size_t slcsize = header.nx * header.ny * psize;
    bufsize = header.nx * header.ny;
    buf = new char[bufsize * psize];
    MPI_File_seek(output, glboffset + slcN * slcsize, MPI_SEEK_SET);
    MPI_File_read(output, buf, slcsize, MPI_CHAR, MPI_STATUS_IGNORE);  
    memcpy(slcdata, buf, sizeof(float) * bufsize);
    delete[] buf;
}

// void MrcStackM::ReadBlock(int start, int end, float *blockdata)
// {
//     MPI_Offset glboffset = sizeof(MRCheader) + header.next;
//     int psize;

//     switch (header.mode)
//     {
//     case MODE_BYTE:
//         psize = sizeof(char);

//         break;

//     case MODE_SHORT:
//         psize = sizeof(short);

//         break;

//     case MODE_FLOAT:
//         psize = sizeof(float);

//         break;

//     default:
//         printf("File type unknown!\n");
//         return;
//     }

//     char *buf;
//     size_t bufsize;
//     int slcN = start;
//     int thickness = end - start;

//     size_t slcsize = header.nx * header.ny * psize;  //一张切片的内存大小


//     bufsize = header.nx * header.ny * thickness;
//     buf = new char[bufsize * psize];

//     MPI_File_seek(mpifile, glboffset + slcN * slcsize, MPI_SEEK_SET);
//     MPI_File_read(mpifile, buf, slcsize * thickness, MPI_CHAR, MPI_STATUS_IGNORE);

//     switch (header.mode)
//     {
//     case MODE_BYTE:
//         for (int i = bufsize; i--;)
//         {
//             blockdata[i] = ((unsigned char *)buf)[i];
//         }

//         break;

//     case MODE_SHORT:
//         for (int i = bufsize; i--;)
//         {
//             blockdata[i] = ((short *)buf)[i];
//         }

//         break;

//     case MODE_FLOAT:
//         memcpy(blockdata, buf, sizeof(float) * bufsize);

//         break;

//     default:
//         printf("File type unknown!\n");
//     }

//     delete[] buf;
//     // return bufsize;
// }

void MrcStackM::ReadBlock(int start, int end, char axis, float *blockdata)
{
    MPI_Offset glboffset = sizeof(MRCheader) + header.next;
    int psize;

    switch (header.mode)
    {
    case MODE_BYTE:
        psize = sizeof(char);

        break;

    case MODE_SHORT:
        psize = sizeof(short);

        break;

    case MODE_FLOAT:
        psize = sizeof(float);

        break;

    default:
        printf("File type unknown!\n");
        return;
    }

    char *buf;
    size_t bufsize;
    int slcN = start;
    int thickness = end - start;

    size_t slcsize = header.nx * header.ny * psize;  //一张切片的内存大小 - Size of a memory slice

    switch (axis)
    {
    case 'z':
    case 'Z':
        bufsize = header.nx * header.ny * thickness;
        buf = new char[bufsize * psize];

        MPI_File_seek(mpifile, glboffset + slcN * slcsize, MPI_SEEK_SET);
        MPI_File_read(mpifile, buf, slcsize * thickness, MPI_CHAR,
                      MPI_STATUS_IGNORE);
        break;

    case 'x':
    case 'X':
        bufsize = header.nz * header.ny * thickness;
        buf = new char[bufsize * psize];
        for (slcN; slcN < end; slcN++)
        {
            char *cur = buf + (slcN - start) * header.nz * header.ny * psize;
            ;

            for (int k = 0; k < header.nz; k++)
            {
                MPI_Offset zoffset = glboffset + k * slcsize;
                for (int j = 0; j < header.ny; j++)
                {
                    MPI_File_seek(mpifile, zoffset + j * header.nx * psize + slcN,
                                  MPI_SEEK_SET);
                    MPI_File_read(mpifile, cur, psize, MPI_CHAR, MPI_STATUS_IGNORE);
                    cur += psize;
                }
            }
        }
        break;

    case 'y':
    case 'Y':
        bufsize = header.nz * header.nx * thickness;  //thickness是steplength - step length
        buf = new char[bufsize * psize];  //buf包含了整个要切的部分 - buf contains the entire part to be cut
        for (slcN; slcN < end; slcN++)
        {
            char *cur = buf + (slcN - start) * header.nz * header.nx * psize;  //将指针移动到buf的特定层起点 - Move the pointer to the starting point of a specific layer of buf

            for (int k = 0; k < header.nz; k++)
            {
                MPI_File_seek(mpifile,
                              glboffset + slcN * header.nx * psize + k * slcsize, //第k层的第slcn列 - The k-th layer of the slcn column
                              MPI_SEEK_SET);
                MPI_File_read(mpifile, cur + k * header.nx * psize, header.nx * psize,
                              MPI_CHAR, MPI_STATUS_IGNORE);   //读取这一列的数据赋给cur - Read the data of this column and assign it to cur
            }
        }
        break;

    default:
        break;
    }

    switch (header.mode)
    {
    case MODE_BYTE:
        for (int i = bufsize; i--;)
        {
            blockdata[i] = ((unsigned char *)buf)[i];
        }

        break;

    case MODE_SHORT:
        for (int i = bufsize; i--;)
        {
            blockdata[i] = ((short *)buf)[i];
        }

        break;

    case MODE_FLOAT:
        memcpy(blockdata, buf, sizeof(float) * bufsize);

        break;

    default:
        printf("File type unknown!\n");
    }

    delete[] buf;
    // return bufsize;
}

bool MrcStackM::WriteHeader()  //函数中的mpifile原来是output - The mpifile in the function was originally output
{
    MPI_File_seek(output, 0, MPI_SEEK_SET);
    int rc = MPI_File_write(output, &header, sizeof(MRCheader), MPI_CHAR,
                   MPI_STATUS_IGNORE);
    if (rc != 0) {
        char error_string[BUFSIZ];
        int length;
        MPI_Error_string(rc, error_string, &length);
        printf("MPI_File_write error: %s\n", error_string);
    }
    return !rc;
}

void MrcStackM::WriteSlice(int slcN, float *slcdata)  //函数中的mpifile原来是output - The mpifile in the function was originally output
{
    int psize = sizeof(float);
    MPI_Offset offset = sizeof(MrcHeader) + header.next;
    size_t slcsize = header.nx * header.ny * psize;
    MPI_File_seek(output, offset + slcN * slcsize, MPI_SEEK_SET);
    MPI_File_write(output, slcdata, header.nx * header.ny, MPI_FLOAT,
                       MPI_STATUS_IGNORE);
}

// void MrcStackM::WriteBlock(int start, int end, float *blockdata)
// {
//     int psize = sizeof(float);
//     size_t del = end - start;

//     MPI_Offset offset = sizeof(MrcHeader) + header.next;
//     MPI_File_seek(output, offset, MPI_SEEK_SET);

//     offset = header.nx * header.ny * psize;
//     MPI_File_seek(output, start * offset, MPI_SEEK_CUR);
//     MPI_File_write(output, blockdata, header.nx * header.ny * del, MPI_FLOAT,
//                        MPI_STATUS_IGNORE);
    
// }

void MrcStackM::WriteBlock(int start, int end, char axis, float *blockdata)
{
    int psize = sizeof(float);
    size_t del = end - start;

    MPI_Offset offset = sizeof(MrcHeader) + header.next;
    MPI_File_seek(output, offset, MPI_SEEK_SET);

    // 	std::cout<<"write block ("<<start<<","<<end<<")"<<std::endl;

    switch (axis)
    {
    case 'x':
    case 'X':
        offset = start * psize;
        MPI_File_seek(output, offset, MPI_SEEK_CUR);
        for (int k = 0; k < header.nz; k++)
        {
            for (int j = 0; j < header.ny; j++)
            {
                MPI_File_write(output, blockdata + j * del + k * del * header.ny, del,
                               MPI_FLOAT, MPI_STATUS_IGNORE);
                offset = (header.nx - del) * psize;
                MPI_File_seek(output, offset, MPI_SEEK_CUR);
            }
        }
        break;

    case 'y':
    case 'Y':
        offset = header.nx * start * psize;
        MPI_File_seek(output, offset, MPI_SEEK_CUR);
        for (int k = 0; k < header.nz; k++)
        {
            MPI_File_write(output, blockdata + k * del * header.nx, del * header.nx,
                           MPI_FLOAT, MPI_STATUS_IGNORE);
            offset = header.nx * (header.ny - del) * psize;
            MPI_File_seek(output, offset, MPI_SEEK_CUR);
        }
        break;

    case 'z':
    case 'Z':
        MPI_Offset offsetZ = header.nx * header.ny * psize;
        MPI_File_seek(output, offset + start * offsetZ, MPI_SEEK_SET);
        MPI_File_write(output, blockdata, header.nx * header.ny * del, MPI_FLOAT,
                       MPI_STATUS_IGNORE);
        break;
    }
}

void MrcStackM::InitializeHeader(int width, int height, int z)
{
    memset(&header, 0, sizeof(MRCheader));

    header.mode = MODE_FLOAT;
    header.mx = width;
    header.my = height;
    header.mz = z;

    header.xlen = 1;
    header.ylen = 1;
    header.zlen = 1;

    header.alpha = 90;
    header.beta = 90;
    header.gamma = 90;

    header.mapc = 1;
    header.mapr = 2;
    header.maps = 3;

    header.amin = 0;
    header.amax = 255;
    header.amean = 128;

    header.creatid = 1000;

    // header.next = 0;
    // header.nint = 2;
    // header.nreal = 1;
    // strncpy(header.cmap, "MAP ", 4);
    // strcpy(header.stamp, "DA\n");
    
}

void MrcStackM::UpdateHeader(bool zero) //函数中的mpifile原来是output - The mpifile in the function was originally output
{
    size_t pxsize = header.nx * header.ny;
    float *slcdata = new float[pxsize];

    printf("Update head!\n");

    float amin = FLT_MAX;
    float amax = FLT_MIN;

    long double amean = 0;

    // read the first slice
    ReadoutputSlice(header.nz * .5, slcdata);
    // printf("根据第%d张图像进行计算\n", (int)header.nz * .5);
    int count = 0;
    for (size_t i = pxsize; i--;)
    {
        if (slcdata[i] < (float)-1e10) 
        {
            if(zero) slcdata[i] = 0.0f;
            continue;
        }
        if (slcdata[i] > amax)
            amax = slcdata[i];
        if (slcdata[i] < amin)
            amin = slcdata[i];
        amean += slcdata[i];
        count++;
    }
    amean /= count;
    WriteSlice(header.nz * .5, slcdata);
    printf("amin is %f, amax is %f, amean is %Lf\n", amin, amax, amean);

    header.amin = amin;
    header.amax = amax;
    header.amean = amean;
    header.nlabl = 0;

    MPI_File_seek(output, 0, MPI_SEEK_SET);
    MPI_File_write(output, &header, sizeof(MRCheader), MPI_CHAR,
                   MPI_STATUS_IGNORE);

    delete[] slcdata;
    printf("Updating finished!\n");
}

void MrcStackM::RotateX(const float *blockdata, int x, int y, int z,
                        float *rotxblock)
{
    size_t slcsize = x * z;
    for (int slcN = 0; slcN < z; slcN++)
    {
        for (int k = 0; k < y; k++)
        {
            const float *cur = blockdata + slcN * x + k * slcsize;
            memcpy(rotxblock, cur, sizeof(float) * x);
            rotxblock += x;
        }
    }
}

void MrcStackM::Close()
{
    if (mpifile != MPI_FILE_NULL)
    {
        MPI_File_close(&mpifile);
        mpifile = MPI_FILE_NULL;
    }
    if (output != MPI_FILE_NULL)
    {
        MPI_File_close(&output);
        output = MPI_FILE_NULL;
    }
}
