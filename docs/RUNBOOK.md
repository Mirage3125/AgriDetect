# 启动与模型说明

## 一键启动 Web

推荐先准备样例数据，再启动 Web：

```bash
python scripts/prepare_demo_data.py
python scripts/prepare_tiny_real_disease_data.py
python scripts/run_web.py
```

浏览器打开：

```text
http://127.0.0.1:5000
```

也可以直接运行：

```bash
python app.py
```

`scripts/run_web.py` 本质上也是加载 `app.py` 里的 Flask 应用，只是作为更明确的启动入口。

## 当前模型文件状态

当前仓库本地状态：

- `models/lstm/lstm_yield.pt`：已存在，产量预测优先使用它。
- `models/lstm/scalers.pkl`：已存在，LSTM 输入/输出归一化使用它。
- `models/yolo/best.pt`：仓库默认不提交；当前本机已通过 PlantDoc smoke 训练生成。
- `models/resnet/resnet18_disease.pth`：仓库默认不提交；当前本机已通过 PlantVillage 子集训练生成。

所以病害识别页面能跑，并不代表已经加载了真实 YOLO/ResNet 权重。代码里做了降级：

- YOLO 权重不存在时，使用 `heuristic-yolo-fallback`，生成轻量级检测结果。
- ResNet 权重不存在时，使用 `heuristic-resnet-fallback`，基于图片颜色统计给出轻量级分类结果。

这保证项目在没有大模型文件时也能稳定展示完整流程，但不能把 fallback 结果当作真实模型精度。

## 训练 ResNet 权重

推荐使用真实 PlantVillage 派生数据集：

```bash
python scripts/prepare_real_disease_classification.py --output data/disease/plantvillage_real_subset --max-per-class 60 --val-ratio 0.2
```

训练真实 baseline：

```bash
python src/disease/train_resnet.py --data data/disease/plantvillage_real_subset --epochs 3 --batch-size 32 --pretrained
```

普通 CPU 想更快跑通，可以冻结 backbone：

```bash
python src/disease/train_resnet.py --data data/disease/plantvillage_real_subset --epochs 3 --batch-size 32 --pretrained --freeze-backbone
```

也可以准备 tiny 样例数据：

```bash
python scripts/prepare_tiny_real_disease_data.py
```

训练 tiny 样例数据：

```bash
python src/disease/train_resnet.py --data data/disease/classification --epochs 5
```

训练后会生成：

```text
models/resnet/resnet18_disease.pth
models/resnet/resnet18_disease.classes.txt
```

真实 baseline 的训练指标会保存到：

```text
outputs/metrics/resnet18_metrics.json
outputs/figures/resnet18_confusion_matrix.png
```

注意：tiny 数据集只有几十张图，只适合验证训练流程，不适合汇报真实精度。正式结果建议使用 `plantvillage_real_subset` 或全量导出数据。

## 训练 YOLO 权重

YOLO 训练需要额外安装 Ultralytics：

```bash
pip install ultralytics
```

准备真实 PlantDoc YOLO 子集：

```bash
python scripts/prepare_plantdoc_yolo.py --output data/disease/plantdoc_yolo_subset --train-limit 200 --val-limit 50
```

快速 smoke 子集：

```bash
python scripts/prepare_plantdoc_yolo.py --output data/disease/plantdoc_yolo_smoke --train-limit 50 --val-limit 10
```

检查数据：

```bash
python src/disease/dataset_check.py --data data/disease/plantdoc_yolo_subset
```

训练：

```bash
python src/disease/train_yolo.py --data data/disease/plantdoc_yolo_subset/data.yaml --epochs 20
```

CPU 快速验证训练链路：

```bash
python src/disease/train_yolo.py --data data/disease/plantdoc_yolo_smoke/data.yaml --epochs 1 --imgsz 416
```

训练脚本会尝试自动复制最佳权重到：

```text
models/yolo/best.pt
```

注意：当前 tiny YOLO 标签规模较小，训练出来只能证明流程跑通。真实检测项目需要更大规模、人工标注框质量更好的 YOLO 数据集。

## 训练 LSTM 权重

默认样例 CSV：

```bash
python scripts/prepare_demo_data.py
```

训练：

```bash
python src/yield_prediction/train_lstm.py --csv data/yield/yield_demo.csv --epochs 30
```

训练后会生成：

```text
models/lstm/lstm_yield.pt
models/lstm/scalers.pkl
outputs/figures/lstm_training_loss.png
outputs/figures/yield_prediction_compare.png
```
