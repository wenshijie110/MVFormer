# Saliency Prediction on Mobile Videos: A Large-scale Dataset and A Transformer Approach
![dataset](https://github.com/wenshijie110/MVFormer/assets/54231028/24281504-fb64-42d3-bda7-1c39b6fd1058)


## 1. Download dataset
The MVS database can be downloaded from [DropBox](https://www.dropbox.com/scl/fi/agy1qts5s9lbzoa5wlw7s/MVS.zip?rlkey=w76j23fdb4k8whg0uiij8djoe&dl=0) or [BiaduPan, code:mvsd](https://pan.baidu.com/s/1cVHbFhSyV0J_dQ3EV0tB2w?pwd=mvsd ). 

The directory structure of the MVS dataset is as follows, 
```
└── MVS  
    ├── Video-Number  
        ├── fixation_maps
        ├── fixations
        ├── frames
        ├── saliency_maps
└── video2frame.py
```
To get each frame of a video, you need to run:
```bash
python video2frame.py 
```

## 2. The training and testing code will be updated soon.

Visualization of predicted saliency maps of our and other compared approaches over two mobile video clips in MVS dataset.

![visualization](https://github.com/wenshijie110/MVFormer/assets/54231028/6ade5405-8148-4bfa-a7f9-2d66fb35e6fd)

# Contact 
If you have any questions, please contact wenshijie@buaa.edu.cn
