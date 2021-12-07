# RUAS

This is the official code for the paper "Learning with Nested Scene Modeling and Cooperative Architecture Search for Low-Light Vision"

A preliminary version of this work has been published in CVPR 2021 [1]. [RUAS](https://openaccess.thecvf.com/content/CVPR2021/html/Liu_Retinex-Inspired_Unrolling_With_Cooperative_Prior_Architecture_Search_for_Low-Light_Image_CVPR_2021_paper.html)
In the conference work, we proposed a new method to integrate the principled optimization unrolling technique with a cooperative prior architecture search strategy for designing an effective yet lightweight low-light image enhancement network. 
In this journal submission, a series of substantial extensions have been made to improve our conference work.

## Environment Preparing
```
python 3.6
pytorch 1.8.0
```

### Testing Enhancement

We provide different models which are trained from different datasets, the models are saved in './weights/'.
*lol.pt* is trained from LOL dataset.
*mit.pt* is trained from MIT5K dataset.
*darkface.pt* is trained from DarkFace dataset.
Finally, run *test_enhancement.py*, the results will be saved in `./result/`
```
python test_enhancement.py 
--data_path           #The folder path of the picture you want to test
'./data/enhance_test_data/lol/test'
--model               #The checkpoint which you want use
'weights/lol.pt'
--save_path            #The save path of the picture processed
./result/
```



### Testing Detection

We provide model which is trained from DarkFACE dataset, please download the model from [Google Drive](https://drive.google.com/file/d/1_9lLdw9yRsgQ6BLFgik-OHt9KWC7M42F/view?usp=sharing) or [Baiduyun](https://pan.baidu.com/s/1fZ5WWxqpyD6CQlY12sIJYA) (Extraction code: ruas) and put it in './weights/'.
Run *test_detection.py*, the results will be saved in `./result/`
```
python test_detection.py 
--data_path           #The folder path of the picture you want to test
'./data/detection_test_data'
--model               #The checkpoint for detection
'weights/detection.pth'
--save_path            #The save path of the picture processed
./result/
```


### Reference

If you find our work useful in your research please consider citing our paper:
```
@inproceedings{liu2021ruas,
title = {Retinex-inspired Unrolling with Cooperative Prior Architecture Search for Low-light Image Enhancement},
author = {Risheng, Liu and Long, Ma and Jiaao, Zhang and Xin, Fan and Zhongxuan, Luo},
booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
year = {2021}
}
```

A great thanks to [DARTS](https://github.com/quark0/darts) for providing the basis for this code.
