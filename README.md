IFIFusion (A Independent Feature Information Fusion Model for Surface Defect Detection)
==========
![image](https://github.com/zhx-hub/IFIFusion/blob/main/img/architecture.jpg)

## Data preparation
Download and extract Magnetic Tile train and val images online.

## Training
To train Net_DeiT-s on MT with 1 gpus for 200 epochs run:
```
python -m --use_env main.py --ck-path /path/to/deit_s --batch-size 224 --data-path /path/to/mt --output_dir /path/to/save
```
## NOTE
The proposed backbone will be allowed to upload by team members when the other network paper is accepted. 

