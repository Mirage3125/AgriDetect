# 本地已准备数据集

最后整理时间：2026-07-08

## 病害图像数据

### PlantVillage 真实分类子集

路径：

- `data/disease/plantvillage_real_subset/`

来源：

- Hugging Face `Hemg/new-plant-diseases-dataset`

当前规模：

- 2,280 张图片
- 38 类
- 训练集 1,850 张
- 验证集 430 张

用途：

- 训练真实病害分类 baseline。
- 当前 ResNet18 权重 `models/resnet/resnet18_disease.pth` 就是基于该子集训练得到。

导出命令：

```bash
python scripts/prepare_real_disease_classification.py --output data/disease/plantvillage_real_subset --max-per-class 60 --val-ratio 0.2
```

### Tiny 真实图像样例数据

路径：

- `data/disease/classification/`
- `data/disease/images/`
- `data/disease/labels/`
- `data/disease/data.yaml`
- `data/disease/DATASET_SOURCES.md`

来源：

- Wikimedia Commons 公开叶片图片，由 `scripts/prepare_tiny_real_disease_data.py` 下载并扩增。

当前规模：

- 分类数据：33 张图片，3 类：`healthy`、`late_blight`、`rust`
- YOLO 数据：33 张图片，配套 YOLO 格式标签

可用命令：

```bash
python src/disease/train_resnet.py --data data/disease/classification --epochs 5
python src/disease/dataset_check.py --data data/disease
python src/disease/train_yolo.py --data data/disease/data.yaml --epochs 20
```

说明：

- 这是一份小型真实图像样例数据集，适合本地烟测和训练流程验证。
- 不建议用它汇报真实模型精度；真实精度需要更大规模 PlantVillage、PlantDoc、Plant Pathology 等数据集。

## 作物产量数据

## PlantDoc YOLO Smoke 检测数据

路径：

- `data/disease/plantdoc_yolo_smoke/`

来源：

- Hugging Face `susnato/plant_disease_detection_processed`
- PlantDoc object detection processed dataset

当前规模：

- 训练图片 50 张
- 验证图片 10 张
- 真实检测框 167 个
- 观测类别 id 22 个

用途：

- 验证真实目标检测数据转换、YOLO 标签格式和训练链路。
- 当前 `models/yolo/best.pt` 是基于该 smoke 子集训练 1 个 epoch 得到。

限制：

- smoke 子集太小，不能展示真实检测精度。
- 当前训练 mAP 为 0，仅说明训练链路跑通。

导出命令：

```bash
python scripts/prepare_plantdoc_yolo.py --output data/disease/plantdoc_yolo_smoke --train-limit 50 --val-limit 10
```

推荐 baseline 子集：

```bash
python scripts/prepare_plantdoc_yolo.py --output data/disease/plantdoc_yolo_subset --train-limit 200 --val-limit 50
```

路径：

- `data/yield/owid_key_crop_yields.csv`

来源：

- Our World in Data Grapher: `https://ourworldindata.org/grapher/key-crop-yields.csv`

字段：

- `Entity`
- `Code`
- `Year`
- `Wheat`
- `Rice`
- `Bananas`
- `Maize`
- `Soybeans`
- `Potatoes`
- `Beans`
- `Peas`
- `Cassava`
- `Cocoa beans`
- `Barley`

用途：

- 可以用于国家/地区级作物产量趋势分析。
- 当前 Web 产量预测默认仍使用 `data/yield/yield_demo.csv`，因为模型输入需要温度、降雨、湿度、土壤、水肥等特征；OWID 数据只有年度作物产量，适合做趋势分析或后续与气象数据拼接。

## PlantDoc 尝试结果

曾尝试下载 PlantDoc GitHub 仓库，但该仓库包含 Windows 不支持的文件名，例如带 `?` 的图片路径，导致 Windows checkout 失败。为了避免留下损坏或空目录，已清理失败残留。

如果后续仍要使用 PlantDoc，建议在 Linux/WSL 环境下载并重命名文件，或使用 Kaggle/Hugging Face 上已经规整过的镜像版本。
