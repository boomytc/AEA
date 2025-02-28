# 音频事件检测系统（AEA）

基于机器学习的音频事件检测系统，可以准确识别音频中的多个事件状态（如停车、慢车、飞行、悬停、转弯等）及其转换时刻。系统采用多进程并行处理，显著提高了训练和检测速度。

## Todo
### Todo List

- [x] 模型训练
  - [x] 使用xgboost进行分类训练模型
  - [ ] 优化模型参数
  - [ ] 添加模型评估指标

- [x] 事件预测功能
  - [x] 实现event_guess.py支持模型选择
  - [ ] 添加批量预测功能
  - [ ] 优化预测速度

- [x] Web界面开发
  - [x] 实现webui.py支持模型选择
  - [ ] 添加更多可视化功能
  - [ ] 优化用户交互体验

## 主要特点

- **高效处理**：多进程并行特征提取，训练和检测速度提升显著
- **多状态检测**：支持多种事件状态检测（停车、慢车、飞行、悬停、转弯等）
- **状态转换识别**：能够检测从一种状态到另一种状态的转换过程
- **高准确率**：平均准确率达到85%以上
- **实时分析**：支持实时状态转换检测
- **置信度评估**：提供预测结果的置信度，有助于判断可靠性
- **用户友好**：提供Web界面，便于可视化分析
- **轻量级依赖**：仅依赖常见Python库

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

## 项目结构

```
AEA/
├── datasets/            # 数据集目录
│   ├── data/            # 音频数据文件
│   └── data_list.txt    # 数据列表（文件路径和标签）
├── models/              # 保存训练好的模型
├── utils/               # 工具函数
│   └── feature_extract.py  # 特征提取模块
├── tools/               # 构建和打包工具
├── README.md            # 项目说明文档
├── requirements.txt     # 依赖库列表
├── train.py             # 模型训练脚本
├── events_guess.py      # 事件预测模块
└── webui.py             # Web界面
```

## 特征提取

系统使用以下音频特征进行事件检测：

1. **MFCC（梅尔频率倒谱系数）**
   - 捕捉音频的音色特征
   - 使用13个系数
   - 包含均值和标准差统计量

2. **梅尔频谱**
   - 表示音频的频率特征
   - 使用40个梅尔频带
   - 包含均值和标准差统计量

3. **光谱对比度**
   - 捕捉音频的动态特征
   - 使用6个频带
   - 包含均值和标准差统计量

4. **零交叉率（ZCR）**
   - 作为音高的替代特征
   - 反映信号的频率变化
   - 包含均值和标准差统计量

5. **RMS能量**
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

状态标签包括：
- 单一状态：`停车`、`慢车`、`飞行`、`悬停`、`转弯`等
- 状态转换：`停车到慢车`、`慢车到飞行`、`飞行到慢车`、`慢车到停车`等

### 2. 训练模型

运行训练脚本：
```bash
python train.py
```

训练过程会自动：
- 从`data_list.txt`加载数据
- 使用多进程并行提取特征
- 训练随机森林分类器
- 保存模型和特征标准化器到`models`目录

训练完成后，模型和特征标准化器将保存在`models`目录下：
- `models/audio_event_model_segments.pkl`：分类模型
- `models/feature_scaler_segments.pkl`：特征标准化器

### 3. 预测音频状态

可以通过两种方式使用训练好的模型：

#### 3.1 使用Python API进行预测

```python
from events_guess import predict_audio_events

# 预测音频事件
audio_file = "test.wav"
events = predict_audio_events(
    audio_file,
    window_size=2.0,        # 分析窗口大小（秒）
    hop_length=1.5,         # 窗口滑动步长（秒）
    confidence_threshold=0.6 # 置信度阈值
)

# 输出预测结果
for event_type, start_time, end_time, confidence in events:
    print(f"事件: {event_type}, 开始: {start_time:.2f}s, 结束: {end_time:.2f}s, 置信度: {confidence:.2%}")
```

#### 3.2 使用Web界面进行可视化分析

启动Web界面：
```bash
streamlit run webui.py
```

Web界面功能：
- 上传音频文件进行分析
- 调整分析参数（窗口大小、滑动步长、置信度阈值）
- 可视化音频波形、梅尔频谱和MFCC特征
- 显示事件检测结果和时间线
- 提供音频播放功能

## 脚本使用示例

本节提供各个脚本的实际使用示例。

### train.py - 模型训练

```bash
# 训练模型
python train.py
```

train.py脚本会：
- 从 datasets/data_list.txt 加载数据
- 使用多进程并行提取音频特征
- 训练随机森林分类器模型
- 保存模型到 models/audio_event_model_segments.pkl
- 保存特征标准化器到 models/feature_scaler_segments.pkl

如需修改数据源或输出路径，需要直接编辑脚本中的相关变量：
```python
# 在prepare_dataset函数调用中修改数据列表文件路径
X_scaled, y, scaler = prepare_dataset("datasets/data_list.txt")

# 在train_model函数中修改模型输出路径
model = train_model(X_scaled, y, model_path="models/audio_event_model_segments.pkl")
```

### events_guess.py - 事件预测

```bash
# 基本用法 - 预测单个音频文件
python events_guess.py path/to/audio.wav
```

events_guess.py接受一个命令行参数：音频文件路径。默认使用以下参数：
- 窗口大小：2.0秒
- 滑动步长：2.0秒
- 置信度阈值：0.6

如需修改这些参数，需要直接调用predict_audio_events函数或编辑main函数：
```python
# 自定义参数示例
from events_guess import predict_audio_events

events = predict_audio_events(
    "path/to/audio.wav",
    window_size=1.0,  # 使用更小的窗口
    hop_length=0.5,   # 使用更小的步长
    confidence_threshold=0.7,    # 修改置信度阈值
    model_path="models/my_model.pkl",
    scaler_path="models/my_scaler.pkl"
)
```

### webui.py - Web界面

```bash
# 启动Web界面
streamlit run webui.py
```

Streamlit可以接受其他标准的配置参数，例如：
```bash
# 指定端口和地址
streamlit run webui.py --server.port 8080 --server.address 0.0.0.0

# 部署模式
streamlit run webui.py --server.headless true
```

Web界面交互步骤:
1. 打开浏览器访问显示的URL（默认为http://localhost:8501）
2. 点击"上传音频文件"按钮选择音频文件
3. 调整侧边栏中的参数（窗口大小、滑动步长、置信度阈值）
4. 点击"开始分析"按钮进行事件检测
5. 查看结果区域中的波形图、频谱图和事件时间线

### 特征提取模块 (utils/feature_extract.py)

在Python代码中使用特征提取模块:

```python
# 导入模块
from utils.feature_extract import extract_features, extract_features_with_segments

# 从文件提取特征
audio_file = "path/to/audio.wav"
features = extract_features(audio_file)
print(f"提取的特征维度: {features.shape}")

# 从内存缓冲区提取特征
import io
import soundfile as sf
import librosa

# 加载音频
y, sr = librosa.load("path/to/audio.wav", sr=22050)

# 创建内存缓冲区
buffer = io.BytesIO()
sf.write(buffer, y, sr, format='WAV')
buffer.seek(0)

# 从缓冲区提取特征
features_from_buffer = extract_features(buffer)

# 分段提取特征
segment_features = extract_features_with_segments(audio_file, segment_duration=1.0)
print(f"分段特征数量: {len(segment_features)}")
```

### 自定义实现示例

如果你想修改现有功能或实现自定义处理流程，可以集成使用项目中的函数：

```python
import sys
sys.path.append("/path/to/AEA")  # 添加项目根目录到路径

# 导入项目中的函数
from utils.feature_extract import extract_features
from events_guess import predict_audio_events, merge_predictions

# 1. 自定义音频处理流程
def process_custom_audio(audio_file, threshold=0.7):
    # 使用自定义参数调用预测函数
    events = predict_audio_events(
        audio_file,
        window_size=1.0,  # 使用更小的窗口
        hop_length=0.5,   # 使用更小的步长
        confidence_threshold=threshold
    )
    
    # 检查预测结果
    if events:
        print(f"检测到 {len(events)} 个事件:")
        for event, start, end, conf in events:
            print(f"- {event}: {start:.2f}s 到 {end:.2f}s (置信度: {conf:.2%})")
        return True
    else:
        print("未检测到显著事件")
        return False

# 2. 批量处理多个音频文件
def batch_process(audio_files):
    results = {}
    for audio_file in audio_files:
        print(f"处理: {audio_file}")
        events = predict_audio_events(audio_file)
        results[audio_file] = events
    return results

# 使用示例
if __name__ == "__main__":
    # 处理单个文件
    process_custom_audio("test.wav")
    
    # 处理多个文件
    files = ["file1.wav", "file2.wav", "file3.wav"]
    results = batch_process(files)
```

## 性能指标

当前模型在测试集上的表现：

| 状态 | 准确率 | 召回率 | F1分数 |
|------|--------|--------|--------|
| 停车 | 100%   | 50%    | 67%    |
| 慢车 | 75%    | 100%   | 86%    |
| 飞行 | 100%   | 100%   | 100%   |
| 悬停 | 95%    | 90%    | 92%    |
| 转弯 | 90%    | 85%    | 87%    |

总体准确率：86%

## 注意事项

1. **音频文件要求**：
   - 支持常见音频格式（wav, mp3, flac等）
   - 建议使用无损格式（如wav）以获得最佳效果
   - 所有音频会自动调整为22050Hz采样率进行处理

2. **内存使用**：
   - 特征提取过程会并行处理
   - 建议为大量音频处理预留足够内存
   - 处理过程使用内存缓冲区而非临时文件，提高效率

3. **模型优化**：
   - 当前模型针对特定场景优化
   - 对于新的音频类型可能需要重新训练
   - 可调整随机森林参数以优化特定场景性能

4. **Web界面限制**：
   - 上传文件大小可能受到Streamlit限制
   - 长音频文件分析可能需要较长时间

## 未来计划

- 增加深度学习模型支持
- 优化长音频处理性能
- 添加批量处理功能
- 改进实时分析能力

## 贡献

欢迎提交Issue和Pull Request以帮助改进项目。

## 许可

MIT License
