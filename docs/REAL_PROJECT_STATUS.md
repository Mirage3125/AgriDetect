# 真实项目状态

更新时间：2026-07-08

## 已完成的真实数据与模型

### 病害分类模型

数据源：

- Hugging Face: `Hemg/new-plant-diseases-dataset`
- 数据集类型：PlantVillage 派生图像分类数据
- 原始规模：70,295 张图像，38 类

本地 baseline 子集：

- 路径：`data/disease/plantvillage_real_subset/`
- 类别数：38
- 训练集：1,850 张
- 验证集：430 张
- 导出命令：

```bash
python scripts/prepare_real_disease_classification.py --output data/disease/plantvillage_real_subset --max-per-class 60 --val-ratio 0.2
```

训练结果：

- 模型：ResNet18 + ImageNet 预训练迁移学习
- 训练轮数：3
- 验证准确率：0.7953
- 权重：`models/resnet/resnet18_disease.pth`
- 类别文件：`models/resnet/resnet18_disease.classes.txt`
- 指标：`outputs/metrics/resnet18_metrics.json`
- 混淆矩阵：`outputs/figures/resnet18_confusion_matrix.png`

训练命令：

```bash
python src/disease/train_resnet.py --data data/disease/plantvillage_real_subset --epochs 3 --batch-size 32 --pretrained
```

普通 CPU 快速训练可使用：

```bash
python src/disease/train_resnet.py --data data/disease/plantvillage_real_subset --epochs 3 --batch-size 32 --pretrained --freeze-backbone
```

全量数据导出可使用：

```bash
python scripts/prepare_real_disease_classification.py --output data/disease/plantvillage_full --full --val-ratio 0.2
```

注意：全量导出约 1.1GB，CPU 全量训练会明显更慢，建议使用 GPU。

## 仍需说明的边界

### YOLO 检测模型

当前项目中 YOLO tiny 数据规模较小，框标注为脚本生成的近似框，只适合验证 YOLO 训练/推理流程。

已经尝试过 PlantDoc 原始 GitHub 仓库，但它包含 Windows 不支持的文件名，完整 checkout 失败。因此当前真实落地成果优先放在 ResNet 病害分类模型上。

已经新增 PlantDoc 处理版转换脚本：

```bash
python scripts/prepare_plantdoc_yolo.py --output data/disease/plantdoc_yolo_subset --train-limit 200 --val-limit 50
```

已验证 smoke 子集：

- 路径：`data/disease/plantdoc_yolo_smoke/`
- 训练图片：50
- 验证图片：10
- 标注框：167
- 观测类别 id：22

已执行 smoke 训练：

```bash
python src/disease/train_yolo.py --data data/disease/plantdoc_yolo_smoke/data.yaml --epochs 1 --imgsz 416
```

训练产物：

- `models/yolo/best.pt`

训练结果：

- mAP50：0.0000
- mAP50-95：0.0000

这说明真实 YOLO 训练链路已打通，但 smoke 子集太小、训练轮数太少，不能作为真实检测精度展示。

数据源：

- Hugging Face: `susnato/plant_disease_detection_processed`
- 来源说明：PlantDoc object detection processed dataset
- 许可证：CC BY 4.0

注意：该数据集全量下载约 5.3GB，且包含较大的预处理张量字段。脚本默认只导出真实框标注子集，适合做 YOLO baseline；全量训练建议在 GPU/Linux 环境执行。

如果要把 YOLO 做到更高水平，需要：

- 使用 Roboflow、PlantDoc 处理版或自建标注数据；
- 获取真实人工框标注；
- 至少几百到几千张图；
- 汇报 Precision、Recall、mAP50、mAP50-95。

### 产量预测模型

当前 LSTM 权重已存在，默认使用 `data/yield/yield_demo.csv`。此外已下载：

- `data/yield/owid_key_crop_yields.csv`

OWID 数据适合做国家/地区级作物产量趋势分析，但缺少温度、降雨、土壤、水肥等特征，不能直接替换当前 LSTM 输入。后续真实化方向是将 OWID 产量数据与 NASA POWER 气象数据按地区和年份拼接。

## 项目展示建议

这个项目目前已经包含两层真实能力：

- 病害分类：使用公开 PlantVillage 派生真实图像数据，完成数据导出、迁移学习训练、评估指标、模型权重和 Web/API 推理闭环。
- 产量预测：已有 LSTM 时序模型和决策解释接口，后续可接入 OWID + NASA POWER 做真实气象产量融合。

YOLO 检测部分当前属于 smoke baseline，重点是展示真实框标注转换和训练链路。若要展示检测精度，需要继续接入更大规模的带人工框标注数据并训练更多轮。
