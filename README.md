# Detecting Human-Object Interactions in Real-Time
This is the repo for the paper named detecting human-object interactions in real-time

## Requirements: hardware
GPU: Titan, Titan Black, Titan X, K20, K40, K80, GTX
## Requirements: software
You should install matlab to validate the result of HOI-RT.

## Preparation
Firstly, you should install cuda, opencv and cudnn. Then set the 1-3 line of the Makefile: 
```
GPU=1
CUDNN=1
OPENCV=1
```
## Installation 

#### 1 Clone the HOI-RT repository
```git clone --recursive git@github.com:ZhongxingPeng/HOI-RT.git ```
#### 2 Build the project
``` 
cd HOI-RT/
make -j8
```
#### 3 Load trained model

## Test
```
./darknet detector test cfg/vcoco.data cfg/yolo-vcoco608.cfg backup/example.jpg
```
## Train
## Valid
