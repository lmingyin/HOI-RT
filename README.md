
# Detecting Human-Object Interactions in Real-Time

This is the repo for the project named detecting human-object interactions in real-time,
see more detail on our [**Tech Report**](https://github.com/lmingyin/HOI-RT/blob/master/Detecting%20Human-Object%20Interactions%20in%20Real-Time.pdf).

## Contents
1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Test](#test)
4. [Train](#train)
5. [Validate](#validate)

## Requirements  
### Hardware  
GPU: Titan, Titan Black, Titan X, K20, K40, K80, GTX
### Software  
You should install matlab to validate the training result of HOI-RT. You should install cuda, opencv and cudnn. Then set the 1-3 line of *Makefile*: 
```
GPU=1
CUDNN=1
OPENCV=1
```
## Installation
1. Clone the HOI-RT repository  
Firstly, make a new folder named **detection** and then
   ```
   cd detection && git clone --recursive git@github.com:lmingyin/HOI-RT.git
   ```  
1. Build the project
   ```
   cd $HOI-RT  && make -j8
   ```  
1. Load the trained model  
   Load [the trained model](https://pan.baidu.com/s/1ZyXGFf2VSXHArigDwMdAoA) which has been trained on vcoco and our labeled dataset.
And put the model in the **detection** folder.
 
## Test  
After successful installation, now you can test HOI-RT.   
```
cd $HOI-RT/
./darknet detector test cfg/vcoco.data cfg/yolo-vcoco608.cfg ../yolo-vcoco608_80000.weights data/kick.jpg 
```
## Train
### Load coco and vcoco datasets     
V-COCO dataset builds off MS COCO, please download [MS-COCO](http://cocodataset.org/#download) images and annotations(coco 2014 is enough), make sure all which in a new folder **coco**, the downloaded extracted image folders like *train2014*, *val2014*, *test2014* should in the new folder **images** which under **coco**, the downloaded extracted annotations like *instances_train2014.json*, *instances_val2014.json* should in the new folder **annotations** which under **coco**.  
1. Clone V-COCO repository (recursively, so as to include COCO API). 
   ```  
   cd coco
   git clone --recursive https://github.com/s-gupta/v-coco.git 
   ```     
1. Pick out annotations from the COCO annotations.      
   ``` 
   cd v-coco 
   python script_pick_annotations.py coco-data/annotations  
   ```
1. Build 
   ```
   cd $VCOCO_DIR/coco/PythonAPI/ && make 
   cd $VCOCO_DIR && make
   ```
1. Show the V-COCO label  
    Copy vcoco_show.py from **script** folder to the folder **v-coco** and run
   ```
   cd v-coco
   python vcoco_show.py
   ```
1. Get more details  
  See more V-COCO introduction in [V-COCO Repository](https://github.com/s-gupta/v-coco).

### Load our dataset
  More training data should be loaded from [our dataset1](https://pan.baidu.com/s/1lRrHTPsKLsNjZd48Iu3v3w) and [our dataset2](https://pan.baidu.com/s/1qK4EOqR1M3XOlI1ueRY5Hg) , then combine the two folder to **RelationDataset** and put it under the folder **detection**. 
### Generate training labels 
1. Make labels from V-COCO dataset   
Copy vcoco_label.py from **script** folder to **v-coco** folder and then
   ```
   cd coco && mkdir filelist
   cd v-coco && python vcoco_label.py
   ```
   Finally, in folder **filelist** will generate a file *trainVCOCO.txt*. And in folder **coco** will outputs a folder named **labels** which contain all training labels.    
1. Make labels from our dataset  
  Copy voc_relation_label.py from the **script** folder to the **detection** folder, 
and then
   ```
   cd detection && python voc_relation_label.py
   ```
   Finally, *trainOurs.txt* will be generated in current folder, and training labels will be generated in every action folder in **RelationDataset**. 

### Merge two training labels 
Copy *trainOurs.txt* to the folder **filelist**. And then
   ```
   cd filelist && cat trainVCOCO.txt trainOurs.txt > train.txt
   ```
### Load the pretrained model 
  Load the [pretrained model](https://pjreddie.com/media/files/darknet19_448.conv.23), and put it in the **detection** folder
### Train the model
Before training, you should set *cfg/vcoco.data*
```
train = Your_Path/coco/filelist/train.txt
```
then use the following command to train the model
```
cd ROI-RT/
make clean && make -j8
./darknet detector train cfg/vcoco.data cfg/yolo-vcoco608.cfg ../darknet19_448.conv.23 
```
## Validate
### Validate action detection on AP<sub>agent</sub>
1. Generate the test labels  
Copy *vcoco_test_action.py* from the **script** folder to **v-coco** folder, and then 
   ```
   cd v-coco
   python vcoco_test_action.py
   ```
   You will get a folder **vcoco_action_valid**. Put it in the **detection** folder.
1. Validate the model on action detection 
   ```
   cd HOI-RT/cfg/
   ``` 
   open *vcoco.data* and set 
   ```
   valid = Your_Path/coco/filelist/vcoco_test.txt
   eval = 
   ```   
   and then validate the model
   ```
   cd HOI-RT/
   ./darknet detector valid cfg/vcoco.data cfg/yolo-vcoco608.cfg backup/yolo-vcoco608_80000.weights
   ```
   in the current folder a folder **results** will be generated, you should put it in the folder **vcoco_action_valid**, and then 
   ```
   cd HOI-RT/matlab
   ```
   run the script *validate_action.m*, you will get the AP<sub>agent</sub> for every action.

### Validate relation detection on AP<sub>role</sub>
1. Generate the test labels  
Copy *vcoco_test_relation.py* from the **script** folder to **v-coco** folder, and then 
   ```
   python vcoco_test_relation.py
   ```
   A folder **vcoco_relation_valid** will be generated, and put it in the folder **detection**.
1. Validate the model on relation detection 
   ```
   cd HOI-RT/cfg/
   ``` 
   open *vcoco.data* and set 
   ```
   valid = Your_Path/coco/filelist/vcoco_test.txt
   eval = relation
   ```   
   and then validate the model
   ```
   cd HOI-RT/
   ./darknet detector valid cfg/vcoco.data cfg/yolo-vcoco608.cfg backup/yolo-vcoco608_80000.weights
   ```
   in current folder a folder **results** will be generated, you should put it in the folder **vcoco_relation_valid**  
   ```
   cd HOI-RT/matlab
   ```
   run the script *validate_relation.m*, you will get the AP<sub>role</sub> for every action. 

 
