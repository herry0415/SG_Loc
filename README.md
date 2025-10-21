# SGLoc
本项目仅用于复现代码记录使用


## Environment
项目环境类似SGLoc和DiffLoc BevDiffLoc 具体看https://github.com/herry0415/BevDiffLoc
- python 3.8.16

- pytorch 1.11.0

- cuda 11.3

```
source install.sh
```

## Data prepare
We use [SPVNAS](https://github.com/mit-han-lab/spvnas) for data preprocessing (just used for training) and generate corresponding planar masks. You need to download the code for SPVNAS and run the [data_prepare.py](code/data_prepare.py) we provided within it.


## 修改代码兼容新的数据集
- data.hercules_lidar.py // data.hercules_radar.py 数据集类加载 
- trainr.py  训练文件
- test.py 测试文件
----

- [train.py](http://train.py) 所有**todo**的
    - **数据集里面hercules_radar.py和hercules.py 的里面的sequence**
    - **服务器id**
    - 导入包要选的radar、lidar
    - **日志文件夹的路径**
    - **序列名 sequence_name**
    - 位姿stats 文件**pose_stats_file  radar和lidar不同**
- [test.py](http://test.py) 所有**todo**的
    - **hercules_radar.py 和 hercules.py** 的里面的sequence
    - **服务器id**
    - 导入包要选的radar、lidar
    - **日志文件夹的路径**
    - 加载**第几轮**的权重**resume_model**
    - 序列名 **sequence_name**
    - 位姿stats 文件**pose_stats_file  radar和lidar不同**

## Run

### train
```
python train.py --dataset_folder xxx
```
- 同类型radar切换只用该序列名+服务器id号码 

### test
```
python test.py --dataset_folder xxx --resume_model checkpoint_epoch_xxx.pth
```
- 同类型radar切换只用该序列名+服务器id号码+权重轮次
## Model zoo

The models of SGLoc on Oxford, QEOxford, and NCLT can be downloaded [here](https://drive.google.com/drive/folders/1FWoNDEsqqwnXmo1iSNmiGlcT0Ox2DM6f?usp=sharing).

## Acknowledgement

 We appreciate the code of [PosePN](https://github.com/PSYZ1234/PosePN) and [STCLoc](https://github.com/PSYZ1234/STCLoc) they shared.

## Citation

```
@inproceedings{li2023sgloc,
  title={SGLoc: Scene Geometry Encoding for Outdoor LiDAR Localization},
  author={Li, Wen and Yu, Shangshu and Wang, Cheng and Hu, Guosheng and Shen, Siqi and Wen, Chenglu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9286--9295},
  year={2023}
}
```
