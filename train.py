import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import librosa
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from feature_extract import extract_features, extract_features_with_segments
import soundfile as sf
from typing import List, Tuple, Dict
import logging
import multiprocessing

def split_transition_audio(audio_file, label):
    """分割转换音频"""
    y, sr = librosa.load(audio_file, sr=None)
    
    # 如果标签包含转换（例如："慢车到飞行"），则分割音频
    if "到" in label:
        states = label.split("到")
        segment_length = len(y) // len(states)
        segments = []
        labels = []
        
        for i, state in enumerate(states):
            start = i * segment_length
            end = (i + 1) * segment_length if i < len(states) - 1 else len(y)
            segment = y[start:end]
            segments.append((segment, sr, state.strip()))
        
        return segments
    else:
        return [(y, sr, label)]

def process_audio_file(args):
    """处理单个音频文件的函数（用于多进程）"""
    audio_file, label = args
    try:
        segments = split_transition_audio(audio_file, label)
        features_list = []
        labels_list = []
        
        for segment, sr, segment_label in segments:
            # 保存临时音频段
            temp_file = f"temp/temp_{multiprocessing.current_process().name}_{os.getpid()}.wav"
            sf.write(temp_file, segment, sr)
            
            # 提取特征
            features = extract_features(temp_file)
            features_list.append(features)
            labels_list.append(segment_label)
            
            # 清理临时文件
            try:
                os.remove(temp_file)
            except:
                pass
                
        return features_list, labels_list
    except Exception as e:
        logging.error(f"处理文件 {audio_file} 时出错: {str(e)}")
        return [], []

def load_dataset_from_file(data_list_file: str, num_processes: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    从文件加载数据集并提取特征
    
    参数:
        data_list_file: 数据列表文件路径
        num_processes: 进程数，默认为CPU核心数-1
    """
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)
    
    print(f"使用 {num_processes} 个进程进行特征提取...")
    
    # 读取数据列表
    with open(data_list_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 准备数据
    data = [(line.strip().split()[0], " ".join(line.strip().split()[1:])) for line in lines]
    
    # 使用多进程处理
    features_all = []
    labels_all = []
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(executor.map(process_audio_file, data))
        
        for features_list, labels_list in results:
            features_all.extend(features_list)
            labels_all.extend(labels_list)
    
    if not features_all:
        raise ValueError("没有成功提取到特征！")
    
    # 转换为numpy数组
    X = np.array(features_all)
    y = np.array(labels_all)
    
    return X, y

def prepare_dataset(data_list_file: str) -> Tuple[np.ndarray, np.ndarray, object]:
    """
    准备数据集
    """
    print("开始加载和处理数据集...")
    
    # 加载数据集
    X, y = load_dataset_from_file(data_list_file)
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 保存标准化器
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/feature_scaler_segments.pkl")
    
    return X_scaled, y, scaler

def train_model(X, y, model_path="models/audio_event_model_segments.pkl"):
    """
    训练随机森林模型并保存
    """
    print("\n开始训练模型...")
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练模型
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    
    # 评估模型
    y_pred = model.predict(X_test)
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 保存模型
    joblib.dump(model, model_path)
    print(f"\n模型已保存到: {model_path}")
    
    return model

if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    
    # 准备数据集
    X_scaled, y, scaler = prepare_dataset("data_list.txt")
    
    # 训练模型
    model = train_model(X_scaled, y)
