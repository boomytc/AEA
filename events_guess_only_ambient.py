import os
import numpy as np
import librosa
import joblib
import soundfile as sf
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor
from utils.feature_extract import extract_features
import time  
import io
import xgboost as xgb
import tempfile
import shutil
from utils.audio_separator import separate_audio

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

def move_files_from_model_dir(track_path, output_dir, model_name, save_file=None):
    """
    将文件从模型子目录移动到目标目录，并删除原始文件夹及其文件
    
    参数:
        track_path: 音频文件路径
        output_dir: 输出目录
        model_name: 模型名称
        save_file: 指定保存的声部（'vocals' 或 'no_vocals'），如果为 None 则保存所有声部
    
    返回:
        str: 分离后的no_vocals文件路径，如果失败则返回None
    """
    model_dir = os.path.join(output_dir, model_name)
    if not os.path.exists(model_dir):
        print(f"模型目录不存在: {model_dir}")
        return None
    
    # 获取音频文件名（不包含路径和扩展名）
    track_name = os.path.splitext(os.path.basename(track_path))[0]
    file_ext = os.path.splitext(track_path)[1]  # 获取原始文件扩展名
    if not file_ext:  # 如果原文件没有扩展名，默认使用.wav
        file_ext = ".wav"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 复制指定声部的文件到目标目录，并重命名
    no_vocals_path = None
    
    # 定义文件匹配和重命名规则
    file_mapping = {
        "vocals": {"pattern": f"{track_name}_vocals", "new_name": f"{track_name}_voice"},
        "no_vocals": {"pattern": f"{track_name}_no_vocals", "new_name": f"{track_name}_ambient"}
    }
    
    # 遍历模型目录中的文件
    for file in os.listdir(model_dir):
        if file.startswith(track_name):
            # 确定文件类型（vocals 或 no_vocals）
            file_type = None
            new_filename = None
            
            # 判断文件类型并确定新文件名
            for stem_type, mapping in file_mapping.items():
                if mapping["pattern"] in file:
                    file_type = stem_type
                    # 保留原始文件扩展名
                    file_ext_current = os.path.splitext(file)[1]
                    new_filename = f"{mapping['new_name']}{file_ext_current}"
                    break
            
            # 如果指定了声部且当前文件不匹配，则跳过
            if save_file and file_type != save_file:
                print(f"  跳过非指定声部文件: {file}")
                continue
            
            # 如果无法确定文件类型，使用原始文件名
            if not new_filename:
                new_filename = file
            
            src_path = os.path.join(model_dir, file)
            dst_path = os.path.join(output_dir, new_filename)
            
            # 复制并重命名文件
            shutil.copy2(src_path, dst_path)  # 复制文件，保留元数据
            print(f"  - 已复制文件: {os.path.basename(dst_path)}")
            
            # 如果是no_vocals文件，记录路径用于返回
            if file_type == "no_vocals" or (save_file == "no_vocals" and file_type == save_file):
                no_vocals_path = dst_path
    
    # 删除模型目录
    try:
        shutil.rmtree(model_dir)
        print(f"  已删除模型目录: {model_dir}")
    except Exception as e:
        print(f"  删除目录失败: {model_dir}, 错误: {e}")
    
    return no_vocals_path

def separate_and_get_ambient(audio_file, temp_dir=None):
    """
    分离音频文件，提取无人声部分
    
    参数:
        audio_file: 音频文件路径
        temp_dir: 临时目录，如果为None则创建新的临时目录
    
    返回:
        str: 分离后的无人声音频文件路径，如果失败则返回原始文件路径
    """
    print(f"开始分离音频文件: {audio_file}")
    
    # 创建临时目录用于存放分离结果
    created_temp_dir = False
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="audio_separate_")
        created_temp_dir = True
    else:
        os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # 调用separate_audio函数分离音频
        model_name = "htdemucs"
        separate_result = separate_audio(
            audio_file,
            output_dir=temp_dir,
            model_name=model_name,
            device="cuda",  # 可以根据可用设备修改为"cpu"
            two_stems="vocals",  # 仅分离人声和伴奏
            verbose=True,
            filename="{track}_{stem}.{ext}"  # 设置输出文件名格式
        )
        
        if not separate_result:
            print(f"音频分离失败，将使用原始音频文件进行事件检测: {audio_file}")
            return audio_file
        
        # 获取分离后的无人声文件
        ambient_file = move_files_from_model_dir(audio_file, temp_dir, model_name, save_file="no_vocals")
        
        if ambient_file:
            print(f"成功提取无人声音频: {ambient_file}")
            return ambient_file
        else:
            print(f"提取无人声音频失败，将使用原始音频文件进行事件检测: {audio_file}")
            return audio_file
    except Exception as e:
        print(f"音频分离过程中出错: {str(e)}")
        print(f"将使用原始音频文件进行事件检测: {audio_file}")
        return audio_file
    finally:
        # 如果是我们创建的临时目录，但分离失败，则删除该目录
        if created_temp_dir and not os.path.exists(os.path.join(temp_dir, os.path.basename(audio_file).replace('.', '_ambient.'))):
            try:
                # 保留成功分离的文件，仅在完全失败时删除临时目录
                if not os.listdir(temp_dir):
                    shutil.rmtree(temp_dir)
                    print(f"已删除临时目录: {temp_dir}")
            except Exception as e:
                print(f"删除临时目录失败: {temp_dir}, 错误: {e}")

def predict_audio_events(
    audio_file: str,
    window_size: float = 2.0,  # 窗口大小（秒）
    hop_length: float = 1.0,    # 窗口滑动步长（秒）
    confidence_threshold: float = 0.55,  # 置信度阈值
    model_path: str = "models/audio_event_model_segments.pkl",
    scaler_path: str = "models/feature_scaler_segments.pkl",
    model_type: str = "rf",  # 模型类型，"rf"表示RandomForest，"xgb"表示XGBoost
    use_ambient_only: bool = True  # 是否只使用无人声部分进行检测
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
    
    # 如果启用了只使用无人声部分，先进行音频分离
    temp_dir = None
    ambient_file = audio_file
    if use_ambient_only:
        print("启用了无人声模式，将先进行音频分离...")
        temp_dir = os.path.join(os.path.dirname(audio_file), "temp_separated_" + os.path.splitext(os.path.basename(audio_file))[0])
        ambient_file = separate_and_get_ambient(audio_file, temp_dir)
    
    # 读取音频文件（可能是原始文件或分离后的无人声文件）
    y, sr = librosa.load(ambient_file, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"开始分析音频文件: {ambient_file}")
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
    parser.add_argument('--confidence', '-c', type=float, default=0.55, help='置信度阈值')
    parser.add_argument('--no-ambient-only', action='store_false', dest='ambient_only', 
                      help='禁用无人声模式（默认启用无人声模式）')
    parser.set_defaults(ambient_only=True)
    
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
        model_type=args.model_type,
        use_ambient_only=args.ambient_only
    )

if __name__ == "__main__":
    main()