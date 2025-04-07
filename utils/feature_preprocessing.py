import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Tuple, Optional

def preprocess_features(
    features: np.ndarray,
    scaler_path: Optional[str] = None,
    save_path: Optional[str] = None,
    fit_scaler: bool = False
) -> Tuple[np.ndarray, object]:
    """
    对特征进行预处理（标准化）
    
    参数:
        features: 需要预处理的特征数组
        scaler_path: 已有标准化器的路径，如果提供则加载已有标准化器
        save_path: 保存标准化器的路径，如果提供则保存标准化器
        fit_scaler: 是否需要拟合标准化器，True表示训练阶段，False表示预测阶段
    
    返回:
        标准化后的特征和标准化器对象
    """
    # 检查是否需要加载已有标准化器
    if scaler_path and os.path.exists(scaler_path) and not fit_scaler:
        print(f"加载标准化器: {scaler_path}")
        scaler = joblib.load(scaler_path)
        features_scaled = scaler.transform(features)
    else:
        # 创建新标准化器
        print("创建新的标准化器...")
        scaler = StandardScaler()
        
        # 拟合并转换
        if fit_scaler:
            print("拟合并转换特征...")
            features_scaled = scaler.fit_transform(features)
        else:
            # 只转换，不拟合（用于预测阶段）
            print("转换特征...")
            features_scaled = scaler.transform(features)
        
        # 保存标准化器
        if save_path:
            print(f"保存标准化器到: {save_path}")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(scaler, save_path)
    
    return features_scaled, scaler
