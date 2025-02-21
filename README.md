# 音频事件检测系统

基于机器学习的音频事件检测系统，可以准确识别音频中的多个事件状态（如停车、慢车、飞行等）及其转换时刻。系统采用多进程并行处理，显著提高了训练和检测速度。

## 主要特点

- 多进程并行特征提取，训练速度快
- 支持多事件状态检测（停车、慢车、飞行等）
- 高准确率（>85%）
- 实时状态转换检测
- 提供置信度评估
- 轻量级依赖

## 系统要求

- Python 3.8+
- Linux/macOS/Windows

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

## 特征提取

系统使用以下音频特征进行事件检测：

1. MFCC（梅尔频率倒谱系数）
   - 捕捉音频的音色特征
   - 使用13个系数
   - 包含均值和标准差统计量

2. 梅尔频谱
   - 表示音频的频率特征
   - 使用40个梅尔频带
   - 包含均值和标准差统计量

3. 光谱对比度
   - 捕捉音频的动态特征
   - 使用6个频带
   - 包含均值和标准差统计量

4. 零交叉率（ZCR）
   - 作为音高的替代特征
   - 反映信号的频率变化
   - 包含均值和标准差统计量

5. RMS能量
   - 作为响度的替代特征
   - 反映信号的能量变化
   - 包含均值和标准差统计量

## 使用方法

### 1. 准备数据

创建一个`data_list.txt`文件，每行包含音频文件路径和对应的标签：
```
/path/to/audio1.wav 停车
/path/to/audio2.wav 慢车到飞行
/path/to/audio3.wav 飞行
```

### 2. 训练模型

运行训练脚本：
```bash
python train.py
```

训练完成后，模型和特征标准化器将保存在`models`目录下：
- `models/audio_event_model_segments.pkl`：分类模型
- `models/feature_scaler_segments.pkl`：特征标准化器

### 3. 预测音频状态

使用训练好的模型进行预测：
```python
from feature_extract import extract_features
import joblib
import numpy as np

# 加载模型和标准化器
model = joblib.load("models/audio_event_model_segments.pkl")
scaler = joblib.load("models/feature_scaler_segments.pkl")

# 提取特征
audio_file = "test.wav"
features = extract_features(audio_file)

# 标准化特征
features_scaled = scaler.transform(features.reshape(1, -1))

# 预测
prediction = model.predict(features_scaled)
probabilities = model.predict_proba(features_scaled)
confidence = np.max(probabilities)

print(f"预测状态: {prediction[0]}")
print(f"置信度: {confidence:.2%}")
```

## 性能指标

当前模型在测试集上的表现：

| 状态 | 准确率 | 召回率 | F1分数 |
|------|--------|--------|--------|
| 停车 | 100%   | 50%    | 67%    |
| 慢车 | 75%    | 100%   | 86%    |
| 飞行 | 100%   | 100%   | 100%   |

总体准确率：86%

## 注意事项

1. 音频文件要求：
   - 支持常见音频格式（wav, mp3, flac等）
   - 建议使用无损格式（如wav）以获得最佳效果
   - 采样率会自动调整为22050Hz

2. 内存使用：
   - 特征提取过程会并行处理
   - 建议为大量音频处理预留足够内存

3. 模型限制：
   - 当前模型针对特定场景优化
   - 对于新的音频类型可能需要重新训练

