# Deep Semantic-aware Proxy Hashing for Multi-label Cross-modal Retrieval [Paper](https://ieeexplore.ieee.org/document/10149001)
This paper is accepted for IEEE Transactions on Circuits and Systems for Video Technology (TCSVT).
If you have any questions please contact hyd199810@163.com.

## Dependencies
We use python to build our code, you need to install those package to run

- pytorch 1.12.1
- sklearn
- tqdm
- pillow

## Training

### Processing dataset
Before training, you need to download the oringal data from [coco](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset)(include 2017 train,val and annotations), nuswide [Google drive](https://drive.google.com/file/d/11w3J98uL_KHWn9j22GeKWc5K_AYM5U3V/view?usp=drive_link), mirflickr25k [Baidu, 提取码:u9e1](https://pan.baidu.com/s/1upgnBNNVfBzMiIET9zPfZQ) or [Google drive](https://drive.google.com/file/d/18oGgziSwhRzKlAjbqNZfj-HuYzbxWYTh/view?usp=sharing) (include mirflickr25k and mirflickr25k_annotations_v080), then use the "data/make_XXX.py" to generate .mat file

After all mat file generated, the dir of `dataset` will like this:
~~~
dataset
├── base.py
├── __init__.py
├── dataloader.py
├── coco
│   ├── caption.mat 
│   ├── index.mat
│   └── label.mat 
├── flickr25k
│   ├── caption.mat
│   ├── index.mat
│   └── label.mat
└── nuswide
    ├── caption.txt  # Notice! It is a txt file!
    ├── index.mat 
    └── label.mat
~~~

### Download CLIP pretrained model
Pretrained model will be found in the 30 lines of [CLIP/clip/clip.py](https://github.com/openai/CLIP/blob/main/clip/clip.py). This code is based on the "ViT-B/32".

You should copy ViT-B-32.pt to this dir.

### Start

After the dataset has been prepared, we could run the follow command to train.
> python main.py --is-train --dataset coco --caption-file caption.mat --index-file index.mat --label-file label.mat --lr 0.001 --output-dim 64 --save-dir ./result/coco/64 --clip-path ./ViT-B-32.pt --batch-size 128 --numclass 80


## Citation
@ARTICLE{10149001,  
  author={Huo, Yadong and Qin, Qibing and Dai, Jiangyan and Wang, Lei and Zhang, Wenfeng and Huang, Lei and Wang, Chengduan},  
  journal={IEEE Transactions on Circuits and Systems for Video Technology},  
  title={Deep Semantic-Aware Proxy Hashing for Multi-Label Cross-Modal Retrieval},  
  year={2024},  
  volume={34},  
  number={1},  
  pages={576-589},  
  doi={[10.1109/TCSVT.2023.3285266](https://ieeexplore.ieee.org/document/10149001)}}  


## Acknowledegements
[DCHMT](https://github.com/kalenforn/DCHMT)
