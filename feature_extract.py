import numpy as np
import librosa
import soundfile as sf
from typing import List, Tuple

def extract_features(audio_file: str, n_mfcc: int = 13, n_mels: int = 40) -> np.ndarray:
    """
    提取音频特征的增强版本
    
    参数:
        audio_file: 音频文件路径
        n_mfcc: MFCC特征数量
        n_mels: 梅尔频谱特征数量
    
    返回:
        包含所有特征的numpy数组
    """
    # 加载音频文件
    y, sr = librosa.load(audio_file, sr=None)
    
    # 1. 提取MFCC特征
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    
    # 2. 提取梅尔频谱特征
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_mean = np.mean(mel_spect, axis=1)
    mel_std = np.std(mel_spect, axis=1)
    
    # 3. 提取色度特征
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)
    
    # 4. 提取光谱对比度
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)
    contrast_std = np.std(contrast, axis=1)
    
    # 5. 提取音调特征
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_mean = np.mean(pitches, axis=1)
    pitch_std = np.std(pitches, axis=1)
    
    # 6. 提取零交叉率
    zero_crossing = librosa.feature.zero_crossing_rate(y)
    zero_mean = np.mean(zero_crossing)
    zero_std = np.std(zero_crossing)
    
    # 7. 提取RMS能量
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)
    
    # 8. 提取频谱中心
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    cent_mean = np.mean(cent)
    cent_std = np.std(cent)
    
    # 9. 提取频谱带宽
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    bandwidth_mean = np.mean(bandwidth)
    bandwidth_std = np.std(bandwidth)
    
    # 10. 提取频谱滚降
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean = np.mean(rolloff)
    rolloff_std = np.std(rolloff)
    
    # 合并所有特征
    features = np.concatenate([
        mfcc_mean, mfcc_std,
        mel_mean, mel_std,
        chroma_mean, chroma_std,
        contrast_mean, contrast_std,
        pitch_mean[:20], pitch_std[:20],  # 只取前20个值以保持固定维度
        [zero_mean, zero_std],
        [rms_mean, rms_std],
        [cent_mean, cent_std],
        [bandwidth_mean, bandwidth_std],
        [rolloff_mean, rolloff_std]
    ])
    
    return features

def extract_features_with_segments(audio_file: str, segment_duration: float = 1.0) -> List[np.ndarray]:
    """
    将音频分段并提取特征
    
    参数:
        audio_file: 音频文件路径
        segment_duration: 每个段的持续时间（秒）
    
    返回:
        每个段的特征列表
    """
    # 加载音频
    y, sr = librosa.load(audio_file, sr=None)
    
    # 计算每个段的样本数
    segment_length = int(segment_duration * sr)
    
    # 分段
    segments = []
    for i in range(0, len(y), segment_length):
        segment = y[i:i + segment_length]
        if len(segment) == segment_length:  # 只保留完整的段
            segments.append(segment)
    
    # 为每个段提取特征
    features_list = []
    for segment in segments:
        # 保存临时音频段
        temp_file = "temp_segment.wav"
        sf.write(temp_file, segment, sr)
        
        # 提取特征
        features = extract_features(temp_file)
        features_list.append(features)
    
    return features_list

def extract_transition_features(audio_file: str, transition_points: List[float]) -> List[Tuple[np.ndarray, str]]:
    """
    在转换点附近提取特征
    
    参数:
        audio_file: 音频文件路径
        transition_points: 转换点时间列表（秒）
    
    返回:
        特征和对应标签的列表
    """
    # 加载音频
    y, sr = librosa.load(audio_file, sr=None)
    
    features_and_labels = []
    window_size = int(1.0 * sr)  # 1秒窗口
    
    for point in transition_points:
        # 转换点的样本索引
        point_idx = int(point * sr)
        
        # 提取转换点前后的音频段
        start_idx = max(0, point_idx - window_size)
        end_idx = min(len(y), point_idx + window_size)
        
        segment = y[start_idx:end_idx]
        if len(segment) < window_size:
            continue
        
        # 保存临时音频段
        temp_file = f"temp_transition_{point}.wav"
        sf.write(temp_file, segment, sr)
        
        # 提取特征
        features = extract_features(temp_file)
        features_and_labels.append((features, f"transition_{point:.1f}"))
    
    return features_and_labels
