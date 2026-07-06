# 智能农业病害识别与产量预测系统

一个面向求职简历和 GitHub 展示的农业 AI 工程项目。项目用 Flask 提供 Web 演示和 API，用 YOLOv8/ResNet18 展示病害检测与分类流程，用 LSTM 展示融合气象、土壤和历史产量特征的产量预测流程。

> 说明：当前仓库默认提供可复现的演示数据和缺权重降级推理，便于本地稳定运行。演示数据为合成数据，不代表真实农业生产结论。

## 功能

- 农作物病害识别：上传 JPG/PNG 图片，选择 YOLOv8 检测或 ResNet18 分类。
- 数据集检查：统计训练/验证/测试图片，检查缺失标签、空标签和损坏图片。
- 产量预测：读取 CSV，完成缺失值处理、异常值裁剪、标准化、LSTM 时间窗口构造和下一期产量预测。
- Web 演示：包含首页、病害识别页、产量预测页。
- API：统一 JSON 响应，适合接口测试和二次开发。

## 系统架构

```text
图片/CSV 数据 -> 数据检查与预处理 -> YOLOv8/ResNet18/LSTM -> Flask API -> Web 页面
```

## 技术栈

Python、PyTorch、torchvision、Ultralytics YOLOv8、pandas、NumPy、scikit-learn、Flask、Matplotlib、HTML、CSS、JavaScript。

## 项目目录

```text
app.py                  Flask 应用入口
config/config.yaml      统一配置
data/                   数据与演示样例
models/                 模型权重放置目录
src/disease/            病害识别模块
src/yield_prediction/   产量预测模块
templates/              Web 页面
static/                 静态资源、上传图片、推理结果
scripts/                环境检查、演示数据、启动脚本
tests/                  API 与模型兜底测试
outputs/                图表、日志、预测输出
```

## 环境要求与安装

推荐使用已有 Conda/Python 环境：

```bash
pip install -r requirements.txt
```

如需单独创建环境：

```bash
conda env create -f environment.yml
conda activate smart-agriculture-ai
```

Ultralytics 仅在真实 YOLOv8 训练/权重推理时需要；没有安装时，Web 演示会自动使用降级检测流程。

## 数据集说明

- `data/yield/yield_demo.csv` 由 `scripts/prepare_demo_data.py` 生成，属于结构化合成演示数据。
- 小型真实病害样例数据可由 `scripts/prepare_tiny_real_disease_data.py` 生成。脚本会从 Wikimedia Commons 下载少量真实叶片/病害图片作为种子图，并扩增为分类和 YOLO 演示数据。
- 真实病害检测数据放在 `data/disease/`，支持常见 YOLO 目录结构：`images/train`、`images/val`、`labels/train`、`labels/val`。
- 真实分类数据放在 `data/disease/classification/<class_name>/`。
- GitHub 仓库默认不提交数据集、模型权重和运行输出，详见 `docs/DATASET.md`。

## 模型说明

- YOLOv8 权重默认路径：`models/yolo/best.pt`
- ResNet18 权重默认路径：`models/resnet/resnet18_disease.pth`
- LSTM 权重默认路径：`models/lstm/lstm_yield.pt`

大权重文件默认被 `.gitignore` 忽略，建议通过 README 说明下载地址，或使用 Git LFS。

## 运行方法

```bash
python scripts/prepare_demo_data.py
python scripts/prepare_tiny_real_disease_data.py
python scripts/run_web.py
```

浏览器打开：`http://127.0.0.1:5000`

## 训练命令

```bash
python src/disease/dataset_check.py --data data/disease
python src/disease/train_yolo.py --data data/disease/data.yaml --epochs 20
python src/disease/train_resnet.py --data data/disease/classification --epochs 5
python src/yield_prediction/train_lstm.py --csv data/yield/yield_demo.csv --epochs 30
```

## 推理命令

```bash
python src/disease/predict_yolo.py --image data/samples/sample_leaf.png
python src/disease/predict_resnet.py --image data/samples/sample_leaf.png
python src/yield_prediction/predict_lstm.py
```

## API 示例

```bash
curl http://127.0.0.1:5000/api/health
curl -X POST http://127.0.0.1:5000/api/yield/predict -H "Content-Type: application/json" -d "{\"use_demo\": true}"
```

病害识别接口：`POST /api/disease/predict`，表单字段为 `image` 和 `method`。

统一响应格式：

```json
{
  "success": true,
  "message": "Prediction completed",
  "data": {}
}
```

## 运行截图

建议上传 GitHub 后补充：

- 首页截图
- 病害识别上传与结果截图
- 产量预测页面截图
- LSTM 训练损失与预测对比图

## 模型评价指标

LSTM 训练后会输出 MAE、RMSE 和 R²，并保存：

- `outputs/figures/lstm_training_loss.png`
- `outputs/figures/yield_prediction_compare.png`

YOLOv8 真实验证可通过 Ultralytics `model.val()` 输出 Precision、Recall、mAP50 和 mAP50-95。

## 项目亮点

- 基于 YOLOv8 和 ResNet18 搭建农作物病害检测与分类流程。
- 使用 LSTM 融合气象、土壤和历史产量特征，实现农作物产量趋势预测。
- 使用 Flask 封装模型推理接口，并实现可交互 Web 演示。
- 完成数据检查、模型训练、指标评估、结果可视化和工程化部署流程。
- 在缺少真实权重或数据时提供可运行降级方案，保证项目可复现。

## 已知限制

- 默认病害识别降级模型是启发式演示，不代表真实模型精度。
- 演示 CSV 是合成数据，仅用于验证功能链路。
- 当前默认使用 CPU，训练速度取决于本机硬件。

## 后续优化方向

- 接入真实农作物病害数据集并训练 YOLOv8/ResNet18。
- 增加模型验证报告导出和批量推理页面。
- 为产量预测接入真实气象和土壤数据源。
- 增加 Docker 部署和 GitHub Actions 自动测试。

## 免责声明

本项目用于学习、简历展示和工程能力演示，不可直接作为农业生产、病害诊断或经济决策依据。

## 简历描述参考

智能农业病害识别与产量预测系统：基于 PyTorch、YOLOv8、ResNet18、LSTM 和 Flask 构建农业 AI Web 应用，实现农作物病害检测/分类、气象土壤时序特征产量预测、数据检查、模型训练、指标评估、可视化和 API 封装；项目提供可复现演示数据和缺权重降级方案，支持本地稳定运行与 GitHub 展示。
