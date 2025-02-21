# 音频事件检测系统

基于机器学习的音频事件检测系统，可以准确识别音频中的多个事件状态（如停车、慢车、飞行等）及其转换时刻。系统采用多进程并行处理，显著提高了训练和检测速度。

## 主要特点

- 多进程并行特征提取，训练速度快
- 支持多事件状态检测（停车、慢车、飞行等）
- 高准确率（>85%）
- 实时状态转换检测
- 提供置信度评估
- 轻量级依赖

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/boomytc/AEA.git
cd AEA
```

2. 创建并激活虚拟环境（推荐）：
```bash
conda create -n AEA python=3.10 -y
conda activate AEA
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 创建必要的目录：
```bash
mkdir temp models
```

## 使用方法

### 1. 准备训练数据

创建包含音频文件路径和对应标签的数据列表文件 `data_list.txt`：
```
/path/to/audio1.wav 停车
/path/to/audio2.wav 慢车
/path/to/audio3.wav 飞行
```

### 2. 训练模型

```bash
python train.py
```

训练完成后，模型将保存在 `models` 目录下。

### 3. 检测事件

```bash
python events_guess.py /path/to/your/audio.wav
```

## 系统架构

- `feature_extract.py`: 特征提取模块
  - MFCC特征
  - 频谱特征
  - 时域特征
  
- `train.py`: 模型训练模块
  - 多进程特征提取
  - RandomForest分类器
  - 模型评估
  
- `events_guess.py`: 事件检测模块
  - 滑动窗口分析
  - 状态转换检测
  - 置信度评估

## 性能指标

- 准确率：86%
- 处理速度：支持实时检测
- 各状态识别效果：
  - 停车：精确率100%
  - 慢车：精确率75%
  - 飞行：精确率100%

## 目录结构

```
.
├── README.md
├── requirements.txt
├── feature_extract.py    # 特征提取模块
├── train.py             # 模型训练模块
├── events_guess.py      # 事件检测模块
├── models/             # 保存训练模型
└── temp/              # 临时文件目录
```

## 注意事项

1. 确保有足够的磁盘空间用于临时文件
2. 建议使用多核CPU以发挥多进程优势
3. 音频文件支持格式：WAV（推荐44.1kHz，16bit）

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request来帮助改进项目。
