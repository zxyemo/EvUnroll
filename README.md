# EvUnroll: Neuromorphic Events based Rolling Shutter Image Correction (CVPR22)

This repo contains the code and dataset for [**EvUnroll: Neuromorphic events based rolling shutter image correction**](https://ci.idm.pku.edu.cn/Zhou_CVPR22a.pdf)  by Xinyu Zhou, Peiqi Duan, Yi Ma, and [Boxin Shi](https://ci.idm.pku.edu.cn/index.htm).

## Examples of RS correction results on Gev-RS dataset 
![example](/figure/RSexample.gif)

## Data
+ **Gev-RS dataset:**  We capture GS frames from a high-speed camera, and simulate RS frames and corresponding event stream. You can download the dataset from [Gev-RS](https://pan.baidu.com/s/1_tZxJBeLaznrI0UomsPh9A?pwd=evun). 
+ **Real data examples:**  We collect real-data examples with an RS-event hybrid camera for testing, you can download it from [RealData](https://pan.baidu.com/s/1tj3X6nfrZFqqNGGxXXvHMQ?pwd=evun)

Gev-RS dataset follows the below directory format:
```
├── path_to_your_dataset_folder/
    ├── Gev-RS/
        ├── train/
            ├── seq1/
                ├── rs_blur/
                    ├── 00000.png
                    ......
                ├── rs_sharp/
                    ├── 00000.png
                    ......
                ├── gt/
                    ├── 00000.png
                    ......
            ├── seq2/
            ......
        ├── test/
            ......
    ├── all_sequence/
        ├── train/
            ├── seq1.avi
            ......
        ├── test/
            ......
    ├── Gev-RS-DVS/
        ├── train/
            ├── seq1/
                ├── seq1.h5
                ├── seq1_events_viz.avi
            ......
        ├── test/
            ......            
```


## Usage
### Dependency
```shell
pip install -r requirements.txt
```
### Test
+ Put the pretrained model in *trained_model/\** .
+ Change the path to the dataset in *util/config.py*.
```
python test.py
```
### Train
The training procedure consists of three steps:

+ Train the synthesis module and flow module seperately.
+ Freeze the weights of above two modules, and train the fusion module.
+ Unfreeze weights of all three modules and finetune the entire network.

To reprocude the training procedure, you might need to slightly modify the training code in different training steps, and 
```
python train.py
``` 

## License
The datasets can be freely used for research and education only. Any commercial use is strictly prohibited.

## Bibtex

```bibtex
@InProceedings{Zhou_2022_CVPR,
    author    = {Zhou, Xinyu and Duan, Peiqi and Ma, Yi and Shi, Boxin},
    title     = {EvUnroll: Neuromorphic Events Based Rolling Shutter Image Correction},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {17775-17784}
}
```

## Contact
If you have any questions, please send an email to zhouxiny@pku.edu.cn
