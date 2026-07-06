# 数据集说明

本仓库默认不上传数据集、模型权重、推理结果和训练输出，只保留目录结构与数据准备脚本。

## 为什么不提交数据集

- 病害图片数据通常体积较大，不适合直接放入普通 Git 仓库。
- 公开数据集可能有各自的授权协议，需要使用者自行确认。
- 模型权重和训练结果属于生成文件，建议使用 Release、网盘、对象存储或 Git LFS 管理。

## 本地数据目录

```text
data/
├── disease/      # YOLO/ResNet 病害数据集，本地放置，不提交
├── samples/      # 本地演示图片，不提交
└── yield/        # 本地产量 CSV，不提交
```

## 生成演示数据

```bash
python scripts/prepare_demo_data.py
```

这会生成：

- `data/samples/sample_leaf.png`
- `data/yield/yield_demo.csv`

## 生成 tiny 真实病害样例数据

```bash
python scripts/prepare_tiny_real_disease_data.py
```

该脚本会尝试下载少量 Wikimedia Commons 公开图片作为种子图，并扩增出小型训练/验证集。生成数据只用于本地功能演示，不随仓库提交。

## 使用自己的分类数据集

```text
data/disease/classification/
├── healthy/
├── late_blight/
└── rust/
```

训练：

```bash
python src/disease/train_resnet.py --data data/disease/classification --epochs 5
```

## 使用自己的 YOLO 数据集

```text
data/disease/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

检查：

```bash
python src/disease/dataset_check.py --data data/disease
```
