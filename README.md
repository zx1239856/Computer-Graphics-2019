# Photo Realistic Rendering - 2019 Computational Graphics

## Basic Functions
+ Multiple parametric objects supported, i.e. plane, sphere, cube and rotary bezier curve 
+ Mesh object (loaded using OBJ file) 
+ Kd-tree accelaration 
+ Axis aligned boundingbox 
+ Nvidia CUDA support (much faster than CPU implementation using OpenMP) 
+ Surface texture mapping 
+ Natural depth of field 
+ ...

## Dependencies
+ OpenCV (for texture image reading)
+ CUDA (if you want to compile GPU version) 

## Build and Execution
```
mkdir build && cd build 
cmake ../src -DBUILD_CUDA=true|false -DBUILD_TEST=true|false ## Enable CUDA or test building according to your need 
make
```

To render, use the following command 
```
./[program-name] [num_spp] [width] [height]
```

