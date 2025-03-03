import os
import numpy as np
import librosa
import joblib
import soundfile as sf
from typing import List, Tuple, Dict, Union, Any
from concurrent.futures import ThreadPoolExecutor
from utils.feature_extract import extract_features
import sys
import time  
import io
import xgboost as xgb

def analyze_audio_segment(segment, sr, model_data, scaler, segment_start_time, model_type="rf"):
    """
    分析音频片段并返回预测结果
    
    参数：
        segment: 音频片段
        sr: 采样率
        model_data: 模型数据，可以是RandomForest模型或XGBoost模型和标签映射的字典
        scaler: 特征标准化器
        segment_start_time: 片段开始时间
        model_type: 模型类型，"rf"表示RandomForest，"xgb"表示XGBoost
    """
    # 使用内存缓冲区代替临时文件
    buffer = io.BytesIO()
    try:
        sf.write(buffer, segment, sr, format='WAV')
        buffer.seek(0)
        
        # 提取特征
        features = extract_features(buffer)
        features = features.reshape(1, -1)
        
        # 标准化特征
        features_scaled = scaler.transform(features)
        
        # 根据模型类型进行预测
        if model_type == "rf":
            # RandomForest模型
            prediction = model_data.predict(features_scaled)[0]
            probabilities = model_data.predict_proba(features_scaled)[0]
            max_prob = np.max(probabilities)
        elif model_type == "xgb":
            # XGBoost模型
            model = model_data['model']
            label_mapping = model_data['label_mapping']
            
            # 创建DMatrix对象
            dmatrix = xgb.DMatrix(features_scaled)
            
            # 预测
            pred_probs = model.predict(dmatrix)
            pred_idx = np.argmax(pred_probs, axis=1)[0]
            # 修复索引错误 - 确保使用整数索引
            prediction = str(label_mapping[int(pred_idx)])  # 明确转换为Python字符串
            max_prob = np.max(pred_probs)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
            
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
    hop_length: float = 1.0,    # 窗口滑动步长（秒）
    confidence_threshold: float = 0.6,  # 置信度阈值
    model_path: str = "models/audio_event_model_segments.pkl",
    scaler_path: str = "models/feature_scaler_segments.pkl",
    model_type: str = "rf"  # 模型类型，"rf"表示RandomForest，"xgb"表示XGBoost
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
        model_type: 模型类型，"rf"表示RandomForest，"xgb"表示XGBoost
    
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
    
    model_data = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # 读取音频文件
    y, sr = librosa.load(audio_file, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"开始分析音频文件: {audio_file}")
    print(f"音频长度: {duration:.2f}秒")
    print(f"使用模型类型: {model_type}")
    
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
            future = executor.submit(analyze_audio_segment, segment, sr, model_data, scaler, segment_start_time, model_type)
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
    import argparse
    
    parser = argparse.ArgumentParser(description='音频事件检测')
    parser.add_argument('audio_file', help='音频文件路径')
    parser.add_argument('--model_type', '-m', choices=['rf', 'xgb'], default='rf', help='模型类型：rf (随机森林) 或 xgb (XGBoost)')
    parser.add_argument('--model_path', help='模型文件路径')
    parser.add_argument('--window_size', '-w', type=float, default=2.0, help='分析窗口大小（秒）')
    parser.add_argument('--hop_length', '-l', type=float, default=1.0, help='窗口滑动步长（秒）')
    parser.add_argument('--confidence', '-c', type=float, default=0.6, help='置信度阈值')
    
    args = parser.parse_args()
    
    # 根据模型类型设置默认模型路径
    model_path = args.model_path
    if model_path is None:
        if args.model_type == 'rf':
            model_path = "models/audio_event_model_segments.pkl"
        else:  # xgb
            model_path = "models/audio_event_model_xgboost.pkl"
    
    predict_audio_events(
        args.audio_file,
        window_size=args.window_size,
        hop_length=args.hop_length,
        confidence_threshold=args.confidence,
        model_path=model_path,
        model_type=args.model_type
    )

if __name__ == "__main__":
    main()