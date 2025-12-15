# Installation & Usage Guide

## 1. System Requirements



- **Linux or Unix-like operating systems**

## 2. Install Dependencies

- **GCC**
  ```bash
  sudo apt install gcc
- **CMake**
    ```bash
  sudo apt install cmake
- **Opencv4.5.5 (C++ version)**
    ```bash
  sudo apt-get install build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
    Download the Opencv4.5.0 installation package (https://opencv.org/releases/).
    unzip opencv-4.5.0.zip
    cd opencv-4.5.0
    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release -DOPENCV_GENERATE_PKGCONFIG=ON -DCMAKE_INSTALL_PREFIX=/usr/local ..
    make -j2
    sudo make install
- **Ceres-Solver2.1 (C++ version)**
    ```bash
    sudo apt-get install libeigen3-dev libatlas-base-dev libsuitesparse-dev
    git clone https://ceres-solver.googlesource.com/ceres-solver
    tar zxf ceres-solver-2.1.0.tar.gz
    mkdir ceres-bin
    cd ceres-bin
    cmake ../ceres-solver-2.1.0
    make -j3
    sudo make install

## 3. Compilation

  ```bash
    mkdir build
    cd ./build
    cmake ..
    make -j36

- **==== Required options ====**

  - **--input (-i):** Specify the input file (The input file extension is .mrc or .st).

    Example: `--input inputfile.st` or `-i inputfile.st`


  - **--initangle (-a):** Specify the initial angle file.

    Example: `--initangle initangle.rawtlt` or `-a initangle.rawtlt`


  - **--diameter (-d):** Specify the marker diameter (pixel).

    Example: `--diameter n` or `-d n` (n is the marker diameter. If -d -1, the software will automatically initialize the diameter)

  - **--txtinput (-x):** Specify the alignment file..

    Example: `-x TS_01.txt`


## 4. Compilation


  ./cal_res \
  -i TS_01_align.mrc \
  -a TS_01.rawtlt \
  -d -1 \
  -x TS_01_align.txt
