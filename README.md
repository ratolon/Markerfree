# Markerfree: GPU-accelerated robust marker-free alignment for high-resolution cryo-electron tomography



## Table of Contents

- [Cryo-ET Tomoalign Toolkit](#cryo-et-tomoalign-toolkit)
  - [Introduction](#introduction)
  - [Table of Contents](#table-of-contents)
  - [Compilation](#compilation)
  - [Dependencies](#dependencies)
  - [Usage](#usage)
  - [Parameters and Configuration](#parameters-and-configuration)
  - [Example](#example)


## Introduction

 A fast, robust and fully automatic software, Markerfree, for efficient and effective cryo-electron tomography (cryo-ET) tilt series alignment.

## Compilation

```bash
cd build
cmake ..
make -j16
```

### Dependencies

- MPI
- CUDA


## Usage

This software provides multiple command-line options for flexible execution of cryo-ET alignment. The basic command structure is:

```bash
./Markerfree [options]
```


## Parameters and Configuration

Below is a list of available options:

`-INPUT(-i) <input_filename>`

Input MRC file for reconstruction.


`-OUTPUT(-o) <output_filename>`

Output MRC filename for results.


`-TILEFILE(-a) <tilt_angle_file>`

Tilt angle filename.


`-GEOMETRY(-g) <seven_integers>`

Geometry information: offset, tilt axis angle, z-axis offset, thickness, projection matching reconstruction thickness, output image downsampling ratio, GPU ID (default: 0 if only one GPU is available).

`-NPROJ (-p) <Number of projections>`

The number of images used during the projection matching phase defaults to 10. Users can adjust this setting as needed, though the default value suffices in most cases.


`-Savemode(-s) <save_format>`

-s 0: Save parameters as txt format (tilt axis angle and translation parameters).
-s 1: Save parameters as xf file format (affine transformation matrix).


`-help(-h)`
Display help information.



## Example
The following example demonstrates alignment using projection matching with a thickness of 300:

```bash
./Markerfree -i /data/BBb.st -o /data/alignresult.mrc -a /data/BBb.rawtlt -g 0,0,0,0,300,1,0 -p 10 -s 0
```
