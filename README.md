# Towards Adaptive Open-set Object Detection via Category-Level Collaboration Knowledge Mining (CCKM)

### This is the official code repository of CCKM. The paper is current under review. Give a big thanks to Dr. Li WuYang with his work [SOMA](https://github.com/CityU-AIM-Group/SOMA). We use it as baseline. The trained pth are  uploaded to [google drive](https://drive.google.com/file/d/1exKp27YTrnwskPgRoHrKc6ffhFEps_fC/view?usp=drive_link).

- ## Environment Preparation

```python
git clone https://github.com/Jandsome/CCKM.git
cd ./CCKM

#Install the project following Deformable DETR
# Linux, CUDA>=9.2, GCC>=5.4
# (ours) CUDA=10.2, GCC=8.4, NVIDIA V100 
# Establish the conda environment

conda create -n CCKM python=3.7 pip
conda activate CCKM
conda install pytorch=1.5.1 torchvision=0.6.1 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt

# 编译
cd ./models/ops
sh ./make.sh

# unit test (should see all checking is True)
python test.py

# NOTE: If you meet the permission denied issue when starting the training
cd ../../ 
chmod -R 777 ./
```

- ##  Download Preparation

###  Download pre-processed datasets (VOC format) from the following links.(Same with SOMA)

|                | (Foggy) Cityscapes                                           | Pascal VOC                                                   | Clipart                                                      | BDD100K (Daytime)                                            |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Official Links | [Imgs](https://www.cityscapes-dataset.com/login/)            | [Imgs+Labels](https://pjreddie.com/projects/pascal-voc-dataset-mirror/) | -                                                            | [Imgs](https://bdd-data.berkeley.edu/)                       |
| Our Links      | [Labels](https://portland-my.sharepoint.com/:u:/g/personal/wuyangli2-c_my_cityu_edu_hk/EVNAjK2JkG9ChREzzqdqJkYBLoZ_VOqkMdhWasN_BETGWw?e=fP9Ae4) | -                                                            | [Imgs+Labels](https://portland-my.sharepoint.com/:u:/g/personal/wuyangli2-c_my_cityu_edu_hk/Edz2YcXHuStIqwM_NA7k8FMBGLeyAGQcSjdSR-vYaVx_vw?e=es6KDW) | [Labels](https://portland-my.sharepoint.com/:u:/g/personal/wuyangli2-c_my_cityu_edu_hk/EQe5cvBEKENIhJuOEIMgmBwBG49OqDidYi3C1eb7vPMWYg?e=RQaddX) |

### Download DINO-pretrained ResNet-50 from this [link](https://portland-my.sharepoint.com/:u:/g/personal/wuyangli2-c_my_cityu_edu_hk/EVnK9IPi91ZPuNmwpeSWGHABqhSFQK52I7xGzroXKeuyzA?e=EnlwgO). We do not utilize the standard ResNet-50 weights to prevent the model from being influenced by the regular ResNet-50 training on the PASCAL VOC dataset or ImageNet dataset.

- ## Change the Path

#### (a) Change the data path as follows.

```
[DATASET_PATH]
└─ Cityscapes
   └─ AOOD_Annotations
   └─ AOOD_Main
      └─ train_source.txt
      └─ train_target.txt
      └─ val_source.txt
      └─ val_target.txt
   └─ leftImg8bit
      └─ train
      └─ val
   └─ leftImg8bit_foggy
      └─ train
      └─ val
└─ bdd_daytime
   └─ Annotations
   └─ ImageSets
   └─ JPEGImages
└─ clipart
   └─ Annotations
   └─ ImageSets
   └─ JPEGImages
└─ VOCdevkit
   └─ VOC2007
   └─ VOC2012
```



#### (b) For bdd100k daytime, put all images into bdd_daytime/JPEGImages/*.jpg.

#### (c) Change the data root in the config files

Replace the DATASET.COCO_PATH in all yaml files in [config](configs) by your data root $DATASET_PATH, e.g., https://github.com/Jandsome/CCKM/blob/main/configs/soma_aood_city_to_foggy_r50.yaml#L22

#### d) Change the path of DINO-pretrained backbone

Replace the backbone loading path:

[SOMA/models/backbone.py](https://github.com/Jandsome/CCKM/blob/main/models/backbone.py#L107)
- ## Training and Testing

```python
CUDA_VISIBLE_DEVICES=0,1 GPUS_PER_NODE=2 ./tools/run_dist_launch.sh 2 python main_multi_eval.py --config_file configs/soma_aood_city_to_foggy_r50.yaml --opts DATASET.AOOD_SETTING 1 OUTPUT_DIR experiments/city_to_foggy/setting1
```
Testing
(1)Change FALSE to TRUE in EVAL in configs.
https://github.com/Jandsome/CCKM/blob/main/configs/soma_aood_city_to_foggy_r50.yaml#L34
(2)Copy the .pth file path to the current config file.
https://github.com/Jandsome/CCKM/blob/main/configs/soma_aood_city_to_foggy_r50.yaml#L76

