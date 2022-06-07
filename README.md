# SegmentationLight

This repo is used for Semantic segmentation which code is easy to understand and effective.

### Generate Model Zoo
- FCNs 
- SegNet
- DeepLab
- HRNet
- Unet
- U2net

A simple benchmark is provided [here](MODEL_ZOO.md), others models(cnn & transformers) will be added if have some time. 

### Code Structure
```
    ├── config                         ------- use for make config class
    |   ├── class_color.py
    │   ├── class_label.py
    │   ├── config.py
    │   ├── __init__.py
    ├── datasets                       ------- datasets class       
    |    ├── coco2017
    |    └── voc2012
    |       ├── color2cls.py
    |       ├── data
    |       ├── dataset.py
    |       ├── __init__.py
    |       ├── make_data_aug.py
    |       ├── make_data.py
    |       ├── process_data.py
    ├── hyparam                        -------- hyparam 
    |   ├── base.yaml
    │   ├── FCNs
    │   ├── HRNet
    │   ├── SegNet
    │   ├── U2Net
    │   └── UNet
    ├── inference_api.py               -------- inference 
    ├── losses                         -------- losses
    |   ├── generatorLoss.py
    │   ├── __init__.py
    │   ├── loss.py
    ├── main.py                        -------- main file
    ├── models                         -------- models factory
    |   ├── DeepLab
    │   ├── FCN
    │   ├── HRNet
    │   ├── __init__.py
    │   ├── model_factory.py
    │   ├── SegNet
    │   ├── U2Net
    │   └── UNet
    ├── post_process.py                -------- post process
    ├── README.md                      -------- readme 
    ├── script                         -------- bash 
    |   ├── FCNs
    │   ├── HRNet
    │   ├── SegNet
    │   └── U2Net
    |       └── train.sh
    ├── train.py                       -------- train file
    ├── utils                          -------- utils 
    │   ├── DataAugments.py
    │   ├── FuseAugments.py
    │   ├── __init__.py
    │   ├── Loss.py
    │   ├── LrSheduler.py
    │   ├── Metirc.py
    │   ├── Optim.py
    │   ├── Summary.py
    │   └── utils.py
``` 

### Clothes Segmentation

- Train
    ```bash
    #!/bin/bash 
    OMP_NUM_THREADS=1
    MKL_NUM_THREADS=1
    export OMP_NUM_THREADS
    export MKL_NUM_THREADS
    cd /data/jiangmingchao/data/code/SegmentationLight;
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -W ignore -m torch.distributed.launch \
    --master_port 2959 \
    --nproc_per_node 8 main.py \
    --hyp /data/jiangmingchao/data/code/SegmentationLight/hyparam/U2Net/baseline_bce_dice_pretrain_320_data.yaml
    ```
- Inference
    ```python
    python inference_api.py
    ```
- Batch Inference
    ```python
    python batch_inference.py
    ```

### Dataset
#### Custom Dataset

1. You need prepare the images & mask pair log, the format like this, each line is a json
```
    {"image_path":"xxxxx/1.jpg", "label_path": "xxxxxx/1.png"}
    {"image_path":"xxxxx/2.jpg", "label_path": "xxxxxx/2.png"}
    {"image_path":"xxxxx/3.jpg", "label_path": "xxxxxx/3.png"}
```
2. make the train.log and val.log, modify the config.yaml
```
CUSTOM_DATASET:
  TRAIN_FILE: "train.log"
  VAL_FILE: "val.log"
  NUM_CLASSES: 2  # used for celoss 
```
3. modify other hyparams in the config for you training.

