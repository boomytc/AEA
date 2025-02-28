import os
import numpy as np
import librosa
import joblib
import soundfile as sf
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor
from utils.feature_extract import extract_features
import sys
import time  
import io

def analyze_audio_segment(segment, sr, model, scaler, segment_start_time):
    """
    分析音频片段并返回预测结果
    """
    # 使用内存缓冲区代替临时文件
    buffer = io.BytesIO()
    try:
        sf.write(buffer, segment, sr, format='WAV')
        buffer.seek(0)
        
        # 提取特征（需要修改feature_extract以支持从内存读取）
        features = extract_features(buffer)
        features = features.reshape(1, -1)
        # 标准化特征
        features_scaled = scaler.transform(features)
        # 预测
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        max_prob = np.max(probabilities)
        return prediction, max_prob
    finally:
        buffer.close()

def merge_predictions(predictions):
    """
    合并相邻的相同预测
    """
    if not predictions:
        return []
    
    merged = []
    current_event = predictions[0]
    current_start = current_event[1]
    current_confidence = [current_event[3]]
    
    for event in predictions[1:]:
        if event[0] == current_event[0]:  # 相同事件
            current_confidence.append(event[3])
        else:  # 不同事件
            # 添加当前事件
            merged.append((
                current_event[0],
                current_start,
                event[1],  # 使用下一个事件的开始时间作为结束时间
                np.mean(current_confidence)
            ))
            # 开始新事件
            current_event = event
            current_start = event[1]
            current_confidence = [event[3]]
    
    # 添加最后一个事件
    merged.append((
        current_event[0],
        current_start,
        current_event[2],
        np.mean(current_confidence)
    ))
    
    return merged

def predict_audio_events(
    audio_file: str,
    window_size: float = 2.0,  # 窗口大小（秒）
    hop_length: float = 2.0,    # 窗口滑动步长（秒）
    confidence_threshold: float = 0.6,  # 置信度阈值
    model_path: str = "models/audio_event_model_segments.pkl",
    scaler_path: str = "models/feature_scaler_segments.pkl"
) -> List[Tuple[str, float, float, float]]:
    """
    对音频文件进行多事件检测
    
    参数：
        audio_file: 音频文件路径
        window_size: 分析窗口大小（秒）
        hop_length: 窗口滑动步长（秒）
        confidence_threshold: 置信度阈值
        model_path: 模型文件路径
        scaler_path: 特征标准化器路径
    
    返回：
        检测到的事件列表，每个事件包含：(事件类型, 开始时间, 结束时间, 置信度)
    """
    process_start_time = time.time()  # 开始计时
    
    # 检查文件是否存在
    if not os.path.exists(audio_file):
        print(f"错误：音频文件不存在 {audio_file}")
        return []
    
    # 加载模型和标准化器
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("错误：模型文件或标准化器文件不存在")
        return []
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # 读取音频文件
    y, sr = librosa.load(audio_file, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"开始分析音频文件: {audio_file}")
    print(f"音频长度: {duration:.2f}秒")
    
    # 计算窗口和步长的样本数
    window_samples = int(window_size * sr)
    hop_samples = int(hop_length * sr)
    
    # 创建线程池
    with ThreadPoolExecutor() as executor:
        # 提交所有分析任务
        futures = []
        for start_sample in range(0, len(y) - window_samples, hop_samples):
            segment_start_time = start_sample / sr
            segment = y[start_sample:start_sample + window_samples]
            future = executor.submit(analyze_audio_segment, segment, sr, model, scaler, segment_start_time)
            futures.append((segment_start_time, future))
        
        # 收集预测结果
        window_predictions = []
        for segment_start_time, future in futures:
            try:
                prediction, confidence = future.result()
                print(f"时间窗口 {segment_start_time:.1f}s - {segment_start_time + window_size:.1f}s:")
                print(f"  预测事件: {prediction}")
                print(f"  置信度: {confidence:.2%}")
                if confidence >= confidence_threshold:
                    window_predictions.append((prediction, segment_start_time, segment_start_time + window_size, confidence))
            except Exception as e:
                print(f"处理时间窗口 {segment_start_time:.1f}s 时出错: {str(e)}")
    
    # 合并相邻的相同预测
    merged_predictions = merge_predictions(window_predictions)
    
    # 输出结果
    print("\n检测到的事件:")
    for event, start, end, confidence in merged_predictions:
        print(f"事件: {event}, 开始时间: {start:.2f}s, 结束时间: {end:.2f}s, 置信度: {confidence:.2%}")
    
    process_end_time = time.time()  # 结束计时
    print(f"\n总执行时间: {process_end_time - process_start_time:.2f}秒")
    
    return merged_predictions

def main():
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        print("请输入音频文件路径")
        sys.exit(1)
    predict_audio_events(audio_file)

if __name__ == "__main__":
    main()