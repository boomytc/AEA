import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import librosa
from concurrent.futures import ProcessPoolExecutor
import xgboost as xgb
from utils.feature_extract import extract_features
import soundfile as sf
from typing import Tuple
import logging
import multiprocessing
import time  
import io

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
    """处理单个音频文件的函数（使用内存缓冲区）"""
    audio_file, label = args
    try:
        segments = split_transition_audio(audio_file, label)
        features_list = []
        labels_list = []
        
        for segment, sr, segment_label in segments:
            # 使用内存缓冲区
            buffer = io.BytesIO()
            try:
                # 将音频数据写入内存缓冲区
                sf.write(buffer, segment, sr, format='WAV')
                buffer.seek(0)  # 重置缓冲区指针到开始位置
                
                # 直接从内存缓冲区提取特征
                features = extract_features(buffer)
                features_list.append(features)
                labels_list.append(segment_label)
            finally:
                buffer.close()
                
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
    start_time = time.time()  # 开始计时
    
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
    
    print(f"\n特征提取总耗时: {time.time() - start_time:.2f}秒")
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

def train_model(X, y, model_path="models/audio_event_model_xgboost.pkl"):
    """
    训练XGBoost模型并保存
    """
    start_time = time.time()  # 开始计时
    print("\n开始训练XGBoost模型...")
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练模型
    # 设置XGBoost参数
    params = {
        'objective': 'multi:softprob',  # 多分类问题
        'eval_metric': 'mlogloss',      # 多分类的损失函数
        'eta': 0.1,                     # 学习率
        'max_depth': 6,                 # 树的最大深度
        'min_child_weight': 1,          # 最小子节点权重
        'subsample': 0.8,               # 样本采样比例
        'colsample_bytree': 0.8,        # 特征采样比例
        'seed': 42,                     # 随机种子
        'nthread': -1,                  # 使用所有CPU核心
        'tree_method': 'hist'           # 使用直方图算法，加速训练
    }
    
    # 获取唯一的标签
    unique_labels = np.unique(y)
    num_classes = len(unique_labels)
    params['num_class'] = num_classes
    
    # 创建DMatrix对象
    dtrain = xgb.DMatrix(X_train, label=[list(unique_labels).index(label) for label in y_train])
    dtest = xgb.DMatrix(X_test, label=[list(unique_labels).index(label) for label in y_test])
    
    # 训练模型
    num_rounds = 100
    model = xgb.train(params, dtrain, num_rounds)
    
    # 评估模型
    y_pred_probs = model.predict(dtest)
    y_pred_indices = np.argmax(y_pred_probs, axis=1)
    y_pred = [unique_labels[idx] for idx in y_pred_indices]
    
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 保存模型和标签映射
    model_data = {
        'model': model,
        'label_mapping': list(unique_labels)
    }
    joblib.dump(model_data, model_path)
    print(f"\n模型已保存到: {model_path}")
    
    print(f"\n模型训练总耗时: {time.time() - start_time:.2f}秒")
    return model, unique_labels

if __name__ == "__main__":
    total_start_time = time.time()  # 总计时开始
    
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    
    # 准备数据集
    X_scaled, y, scaler = prepare_dataset("datasets/data_list.txt")
    
    # 训练模型
    model, unique_labels = train_model(X_scaled, y)
    
    print(f"\n整个训练流程总耗时: {time.time() - total_start_time:.2f}秒")
