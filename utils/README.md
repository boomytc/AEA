# AEA 工具模块 (Utils)

本目录包含音频事件分析系统 (Audio Event Analyzer) 的核心工具函数和模块，提供音频处理、特征提取和音频分离等基础功能。

## 模块概述

### 1. `feature_extract.py`

提供音频特征提取功能，支持从文件路径或内存缓冲区提取多种音频特征。

**主要功能：**
- 提取MFCC特征（梅尔频率倒谱系数）
- 提取梅尔频谱特征
- 提取光谱对比度特征
- 提取零交叉率
- 提取RMS能量
- 支持音频分段特征提取

**主要函数：**
- `extract_features(audio_input, n_mfcc=13, n_mels=40)` - 从单个音频提取特征
- `extract_features_with_segments(audio_file, segment_duration=1.0)` - 将音频分段并提取特征

### 2. `audio_separator.py`

基于Demucs库的音频分离工具，用于将音频分离为不同的声部（如人声、鼓声、贝斯等）。

**主要功能：**
- 支持多种预训练模型
- 可配置的分离参数
- 支持CPU和CUDA加速
- 灵活的输出格式和命名选项

**主要函数：**
- `separate_audio(track_path, ...)` - 使用Demucs进行音频分离
- 命令行接口，支持通过参数调整分离选项

## 使用示例

### 特征提取

```python
from utils.feature_extract import extract_features

# 从文件提取特征
features = extract_features("path/to/audio.wav")
print(f"特征维度: {features.shape}")

# 自定义MFCC和梅尔频谱特征数量
features = extract_features("path/to/audio.wav", n_mfcc=20, n_mels=60)
```

### 音频分离

```python
from utils.audio_separator import separate_audio

# 基本用法
separate_audio("path/to/audio.mp3")

# 高级用法
separate_audio(
    "path/to/audio.mp3",
    output_dir="my_separated",
    model_name="htdemucs",
    device="cuda",
    two_stems="vocals",  # 仅分离人声和伴奏
    verbose=True
)
```

## 与系统集成

这些工具模块被AEA系统的其他组件使用：

1. `events_guess.py` 使用 `feature_extract.py` 提取特征用于音频事件检测
2. `gui.py` 通过这些工具提供图形界面的音频分析功能
3. Web界面组件使用这些工具进行在线音频分析

## 技术细节

- 特征提取基于librosa库实现
- 音频分离基于Demucs库实现
- 支持多种音频格式（WAV, MP3, FLAC等）
- 针对性能进行了优化，支持多线程处理
- 支持内存缓冲区处理，减少磁盘IO

## 最近更新

- 优化了XGBoost模型的特征处理，解决了numpy字符串类型转换问题
- 增强了错误处理和异常信息展示
- 提高了特征提取的性能和内存效率