# MODEL ZOO
### VOC2012
|Models|BatchSize|GPUs|HyParameters|CropSize|GFLOPs|PA|MPA|MIOU|FWIOU|Weights|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|FCNS_vgg16_32s|8|1|[fcn_vgg16_32s](/data/jiangmingchao/data/code/SegmentationLight/hyparam/FCNs/fcn_vgg16_32s.yaml)|512|-|82.9%|50.6%|39.3%|72.2%|/data/jiangmingchao/data/AICKPT/Seg/FCNs/fcn32_vgg16_gpux1_512/checkpoints/best_ckpt_losses_0.5197840489365242_miou_0.39294091458955355.pth|
|FCNS_vgg16_16s|8|1|[fcn_vgg16_16s](/data/jiangmingchao/data/code/SegmentationLight/hyparam/FCNs/fcn_vgg16_16s.yaml)|512|-|86%|57.8%|44.9%|77.1%|/data/jiangmingchao/data/AICKPT/Seg/FCNs/fcn16_vgg16_gpux1_512/checkpoints/best_ckpt_losses_0.4749274262851411_miou_0.449459438309746.pth|
|FCNS_vgg16_8s|16|1|[fcn_vgg16_8s](hyparam/FCNs/fcn_vgg16_8s.yaml)|512|-|87.3%|59.1%|48.1%|78.6%|/data/jiangmingchao/data/AICKPT/Seg/FCNs/fcn8_vgg16_gpux1_512/checkpoints/best_ckpt_losses_0.43843183134283337_miou_0.48058054398118843.pth|
|FCNS_r50_8s|8|1|[fcn_r50_8s](hyparam/FCNs/fcn_resnet50_8s.yaml)|512|-|90.1%|69.9%|58.7%|82.8%|/data/jiangmingchao/data/AICKPT/Seg/FCNs/fcn8_r50_gpux1_512/checkpoints/best_ckpt_losses_0.31139873512662375_miou_0.5870208409364805.pth|
|FCNS_r101_8s|8|1|[fcn_r101_8s](hyparam/FCNs/fcn_resnet101_8s.yaml)|512|-|90.8%|71.7%|61.1%|83.9%|/data/jiangmingchao/data/AICKPT/Seg/FCNs/fcn8_r101_gpux1_512/checkpoints/best_ckpt_losses_0.28888792892570025_miou_0.6112685436040116.pth|
|FCNS_r152_8s|8|1|[fcn_r152_8s](hyparam/FCNs/fcn_resnet152_8s.yaml)|512|-|91.5%|74.1%|63.6%|85.0%|/data/jiangmingchao/data/AICKPT/Seg/FCNs/fcn8_r152_gpux1_512/checkpoints/best_ckpt_losses_0.262765350823219_miou_0.6356795263547104.pth|
|FCNS_mbv2_8s|8|1|[fcn_mbv2_8s](hyparam/FCNs/fcn_mbv2_8s.yaml)|512|-|87.5%|56.7%|47.6%|78.7%|/data/jiangmingchao/data/AICKPT/Seg/FCNs/fcn8_mbv2_gpux1_512/checkpoints/best_ckpt_losses_0.47803407727362035_miou_0.47626235483478524.pth|
|FCNS_mbv3_8s|8|1|[fcn_mbv3_8s](hyparam/FCNs/fcn_mbv3_8s.yaml)|512|-|87.4%|55.4%|46.4%|78.5%|/data/jiangmingchao/data/AICKPT/Seg/FCNs/fcn8_mbv3_gpux1_512/checkpoints/best_ckpt_losses_0.4714071523029726_miou_0.4640446924410685.pth|
|SegNet_vgg16_upsample|16|1|[segnet_vgg16_up](hyparam/SegNet/seg_vgg16.yaml)|512|-|87.7%|59.0%|48.6%|79.3%|/data/jiangmingchao/data/AICKPT/Seg/SegNet/segnet_vgg16_gpux1_512/checkpoints/best_ckpt_losses_0.443921719114859_miou_0.4860254266784411.pth|
|SegNet_vgg16_pool|16|1|[segnet_vgg16_pool](hyparam/SegNet/seg_vgg16_pool.yaml)|512|-|87.0%|54.5%|44.4%|78.2%|/data/jiangmingchao/data/AICKPT/Seg/SegNet/segnet_vgg16_gpux1_512_pool/checkpoints/best_ckpt_losses_0.5337066383479716_miou_0.4440478477271453.pth|
|SegNet_R50_up|8|1|[segnet_r50_up](hyparam/SegNet/seg_r50.yaml)|512|-|88.5%|60.8%|51.1%|80.2%|/data/jiangmingchao/data/AICKPT/Seg/SegNet/segnet_r50_gpux1_512/checkpoints/best_ckpt_losses_0.3944489204294079_miou_0.5108976440062539.pth|
|SegNet_mobilenetv2_up|8|1|[segnet_mobilenetv2_up](hyparam/SegNet/seg_mbv2.yaml)|512|-|88.3%|59.5%|49.7%|79.9%|/data/jiangmingchao/data/AICKPT/Seg/SegNet/segnet_mbv2_gpux1_512/checkpoints/best_ckpt_losses_0.42041630568085137_miou_0.4966990811194377.pth|
|UNet_resnet50|8|2|[unet_resnet50](hyparam/UNet/unet_resnet50.yaml)|512|-|89.6%|64.2%|53.8%|82.0%|/data/jiangmingchao/data/AICKPT/Seg/UNet/unet_resnet50/checkpoints/best_ckpt_losses_0.412014008632728_miou_0.5454743759021599.pth|

