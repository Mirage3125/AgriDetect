# AgriDetect：智能农业病害识别与产量预测系统

AgriDetect 是一个面向农业场景的 AI Web 项目，覆盖作物病害图像识别、目标检测训练链路、作物产量预测、风险解释和 Flask API 服务。项目将数据准备、模型训练、评估指标、模型推理、Web 交互和工程文档串成一条可复现流水线。

## 项目亮点

- **真实病害分类训练闭环**：基于 Hugging Face `Hemg/new-plant-diseases-dataset` 导出 PlantVillage 派生数据子集，使用 ResNet18 + ImageNet 预训练完成 38 类病害分类 baseline。
- **目标检测链路可复现**：支持将 PlantDoc 处理版检测数据转换为 YOLO 格式，并已完成 YOLOv8n smoke 训练，生成 `models/yolo/best.pt`。
- **产量预测与决策解释**：使用 LSTM/回归 fallback 预测作物产量，并返回风险等级、近期均值对比、置信区间、敏感性分析和管理建议。
- **Web + API 一体化**：Flask 提供病害识别、产量预测和健康检查接口，前端展示图片结果、模型引擎、置信度、风险和建议。
- **工程化配套**：包含数据检查、训练脚本、指标输出、混淆矩阵、测试用例和运行文档。

## 当前模型状态

| 模块 | 模型/数据 | 当前状态 | 说明 |
| --- | --- | --- | --- |
| 病害分类 | ResNet18 | 已训练 | 38 类，验证准确率约 `79.53%` |
| 病害检测 | YOLOv8n | 已完成 smoke 训练 | 基于 PlantDoc 真实框小子集，mAP 仍为 0，仅证明检测训练链路 |
| 产量预测 | LSTM | 已有本地权重 | 支持 Web/API 预测和风险解释 |

本地模型默认路径：

```text
models/resnet/resnet18_disease.pth
models/resnet/resnet18_disease.classes.txt
models/yolo/best.pt
models/lstm/lstm_yield.pt
models/lstm/scalers.pkl
```

> 注意：模型权重和数据集默认被 `.gitignore` 忽略，不建议直接提交到普通 Git 仓库。上传 GitHub 时可以通过 Release、网盘、对象存储或 Git LFS 管理权重。

## 技术栈

- Python, Flask
- PyTorch, torchvision
- Ultralytics YOLOv8
- pandas, NumPy, scikit-learn
- Hugging Face datasets, pyarrow
- Matplotlib
- HTML, CSS, JavaScript
- pytest

## 快速启动

安装依赖：

```bash
pip install -r requirements.txt
```

准备基础样例数据：

```bash
python scripts/prepare_demo_data.py
python scripts/prepare_tiny_real_disease_data.py
```

启动 Web：

```bash
python scripts/run_web.py
```

浏览器打开：

```text
http://127.0.0.1:5000
```

健康检查：

```bash
curl http://127.0.0.1:5000/api/health
```

返回示例：

```json
{
  "success": true,
  "message": "Service is healthy",
  "data": {
    "torch_available": true,
    "yolo_weights_exists": true,
    "resnet_weights_exists": true,
    "lstm_weights_exists": true
  }
}
```

## 页面功能

- 首页：项目入口与模块导航。
- 病害识别：上传叶片图片，选择 YOLOv8 检测或 ResNet18 分类，展示原图、结果图、类别、置信度、推理引擎和处置建议。
- 产量预测：输入气象、土壤、水肥、种植面积和历史产量特征，输出预测产量、风险等级、置信区间、敏感性分析和管理建议。

## 数据集与训练

### 1. ResNet18 病害分类

数据源：

- Hugging Face: `Hemg/new-plant-diseases-dataset`
- 原始规模：70,295 张图像，38 类
- 本项目 baseline 子集：2,280 张图像，训练集 1,850 张，验证集 430 张

导出数据：

```bash
python scripts/prepare_real_disease_classification.py --output data/disease/plantvillage_real_subset --max-per-class 60 --val-ratio 0.2
```

训练：

```bash
python src/disease/train_resnet.py --data data/disease/plantvillage_real_subset --epochs 3 --batch-size 32 --pretrained
```

CPU 快速训练：

```bash
python src/disease/train_resnet.py --data data/disease/plantvillage_real_subset --epochs 3 --batch-size 32 --pretrained --freeze-backbone
```

训练产物：

```text
models/resnet/resnet18_disease.pth
models/resnet/resnet18_disease.classes.txt
outputs/metrics/resnet18_metrics.json
outputs/figures/resnet18_confusion_matrix.png
```

已完成 baseline 指标：

```text
epoch=1 loss=1.3893 train_acc=0.6124 val_acc=0.6977
epoch=2 loss=0.5231 train_acc=0.8395 val_acc=0.7651
epoch=3 loss=0.3295 train_acc=0.9027 val_acc=0.7953
```

### 2. YOLOv8 病害检测

数据源：

- Hugging Face: `susnato/plant_disease_detection_processed`
- 来源：PlantDoc object detection processed dataset
- 许可证：CC BY 4.0

导出 smoke 子集：

```bash
python scripts/prepare_plantdoc_yolo.py --output data/disease/plantdoc_yolo_smoke --train-limit 50 --val-limit 10
```

导出更大的 baseline 子集：

```bash
python scripts/prepare_plantdoc_yolo.py --output data/disease/plantdoc_yolo_subset --train-limit 200 --val-limit 50
```

检查 YOLO 数据：

```bash
python src/disease/dataset_check.py --data data/disease/plantdoc_yolo_smoke
```

训练 smoke baseline：

```bash
python src/disease/train_yolo.py --data data/disease/plantdoc_yolo_smoke/data.yaml --epochs 1 --imgsz 416
```

训练脚本会自动复制最佳权重到：

```text
models/yolo/best.pt
```

当前 YOLO 说明：

- smoke 子集：训练 50 张，验证 10 张，真实检测框 167 个，观测类别 id 22 个。
- 当前 mAP50 和 mAP50-95 为 0，说明数据量和训练轮数不足，不适合作为真实检测精度展示。
- YOLO 模块主要展示真实检测数据转换、训练入口、权重管理和 Web 推理链路。

### 3. LSTM 产量预测

准备样例 CSV：

```bash
python scripts/prepare_demo_data.py
```

训练：

```bash
python src/yield_prediction/train_lstm.py --csv data/yield/yield_demo.csv --epochs 30
```

训练产物：

```text
models/lstm/lstm_yield.pt
models/lstm/scalers.pkl
outputs/figures/lstm_training_loss.png
outputs/figures/yield_prediction_compare.png
```

项目还下载了 `data/yield/owid_key_crop_yields.csv`，可用于国家/地区级作物产量趋势分析。由于 OWID 数据缺少温度、降雨、土壤、水肥等特征，当前不直接替换 LSTM 输入。

## API 示例

产量预测：

```bash
curl -X POST http://127.0.0.1:5000/api/yield/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"use_demo\": true}"
```

病害识别：

```bash
curl -X POST http://127.0.0.1:5000/api/disease/predict ^
  -F "method=resnet" ^
  -F "image=@data/samples/sample_leaf.png"
```

统一响应格式：

```json
{
  "success": true,
  "message": "Prediction completed",
  "data": {}
}
```

## 项目结构

```text
app.py                         Flask 应用入口
config/config.yaml             项目配置
data/                          本地数据目录，不建议提交
docs/                          数据、训练、运行和项目状态说明
models/                        本地模型权重目录，不建议提交
outputs/                       指标、图表和预测输出
scripts/                       数据准备和启动脚本
src/common/                    配置、日志等通用模块
src/disease/                   病害分类/检测训练与推理
src/yield_prediction/          产量预测训练、推理和解释分析
static/                        前端静态资源
templates/                     Flask 页面模板
tests/                         API 与模型流程测试
```

## 测试

```bash
python -m pytest
```

当前验证结果：

```text
4 passed
```

测试会覆盖：

- `/api/health`
- 病害识别接口
- 产量预测接口
- 模型推理/fallback 流程

## 文档

- `docs/RUNBOOK.md`：启动、训练和模型文件说明。
- `docs/DOWNLOADED_DATASETS.md`：已准备数据集说明。
- `docs/REAL_PROJECT_STATUS.md`：真实数据、模型、指标和边界。
- `docs/DATASET.md`：数据目录和数据集管理说明。

## 已知边界

- ResNet18 分类 baseline 已完成真实训练，但数据子集规模有限，指标不能等同于生产环境精度。
- YOLOv8 已打通真实框标注训练链路，但当前 smoke 权重不具备可用检测精度。
- LSTM 默认使用样例 CSV，真实产量预测需要进一步融合 OWID、NASA POWER、土壤和区域管理数据。
- 本项目用于学习、研究和工程能力说明，不应直接作为农业生产诊断或经济决策依据。

## 后续计划

- 使用更大 PlantDoc/Roboflow 标注集训练 YOLOv8，并汇报 Precision、Recall、mAP50、mAP50-95。
- 将 OWID 产量数据与 NASA POWER 气象数据按地区和年份拼接，构建真实产量预测数据集。
- 增加批量推理、模型版本管理和导出报告页面。
- 增加 Dockerfile 与 GitHub Actions 自动测试。
