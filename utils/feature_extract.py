import numpy as np
import librosa
import os
import sys
from typing import List, Union
import soundfile as sf
import time  
import io

def extract_features(audio_input: Union[str, io.BytesIO], n_mfcc: int = 13, n_mels: int = 40) -> np.ndarray:
    """
    提取音频特征，支持文件路径或内存缓冲区输入
    
    参数:
        audio_input: 音频文件路径或BytesIO对象
        n_mfcc: MFCC特征数量
        n_mels: 梅尔频谱特征数量
    
    返回:
        包含所有特征的numpy数组
    """
    start_time = time.time()
    try:
        # 根据输入类型选择不同的加载方式
        if isinstance(audio_input, str):
            y, sr = librosa.load(audio_input, sr=22050, mono=True)
        else:
            # 从内存缓冲区读取
            y, sr = sf.read(audio_input)
            # 重采样到22050Hz
            if sr != 22050:
                y = librosa.resample(y, orig_sr=sr, target_sr=22050)
                sr = 22050

        if len(y) == 0:
            raise ValueError("音频数据为空或无法读取")

        # 使用较大的hop_length来减少计算量
        hop_length = 1024
        n_fft = 2048

        # 1. 提取MFCC特征
        # print("提取MFCC特征...")
        mfccs = librosa.feature.mfcc(
            y=y, 
            sr=sr, 
            n_mfcc=n_mfcc,
            hop_length=hop_length,
            n_fft=n_fft
        )
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        
        # 2. 提取梅尔频谱特征
        # print("提取梅尔频谱特征...")
        mel_spect = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=n_mels,
            hop_length=hop_length,
            n_fft=n_fft
        )
        mel_mean = np.mean(mel_spect, axis=1)
        mel_std = np.std(mel_spect, axis=1)
        
        # 3. 提取光谱对比度特征
        # print("提取光谱对比度特征...")
        contrast = librosa.feature.spectral_contrast(
            y=y, 
            sr=sr,
            n_bands=6,
            hop_length=hop_length,
            n_fft=n_fft
        )
        contrast_mean = np.mean(contrast, axis=1)
        contrast_std = np.std(contrast, axis=1)

        # 4. 提取零交叉率（作为音高的替代特征）
        # print("提取零交叉率...")
        zcr = librosa.feature.zero_crossing_rate(
            y,
            hop_length=hop_length
        )
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        
        # 5. 提取RMS能量（作为响度的替代特征）
        # print("提取RMS能量...")
        rms = librosa.feature.rms(
            y=y,
            hop_length=hop_length
        )
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        
        # 合并所有特征
        features = np.concatenate([
            mfcc_mean, mfcc_std,
            mel_mean, mel_std,
            contrast_mean, contrast_std,
            [zcr_mean, zcr_std, rms_mean, rms_std]
        ])
        
        # print("特征提取完成")
        print(f"特征提取耗时: {time.time() - start_time:.2f}秒")
        return features
    except Exception as e:
        print(f"特征提取过程中出错: {str(e)}")
        raise

def extract_features_with_segments(audio_file: str, segment_duration: float = 1.0) -> List[np.ndarray]:
    """
    将音频分段并提取特征
    
    参数:
        audio_file: 音频文件路径
        segment_duration: 每个段的持续时间（秒）
    
    返回:
        每个段的特征列表
    """
    start_time = time.time()  # 开始计时
    try:
        # 加载音频
        y, sr = librosa.load(audio_file, sr=None, mono=True)
        
        if len(y) == 0:
            raise ValueError("音频文件为空或无法读取")
        
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
        
        for i, segment in enumerate(segments):
            # 创建临时目录（如果不存在）
            os.makedirs("temp", exist_ok=True)
            # 保存临时音频段
            temp_file = os.path.join("temp", f"temp_segment_{i}.wav")
            sf.write(temp_file, segment, sr)
            
            try:
                # 提取特征
                features = extract_features(temp_file)
                features_list.append(features)
            except Exception as e:
                print(f"处理段 {i} 时出错: {str(e)}")
                continue
            finally:
                # 清理临时文件
                try:
                    os.remove(temp_file)
                except:
                    pass
        
        print(f"分段特征提取总耗时: {time.time() - start_time:.2f}秒")
        return features_list
    except Exception as e:
        print(f"分段特征提取过程中出错: {str(e)}")
        raise

if __name__ == "__main__":
    total_start_time = time.time()  # 总计时开始
    try:
        if len(sys.argv) > 1:
            audio_file = sys.argv[1]
        else:
            print("请输入音频文件路径")
            sys.exit(1)
        
        if not os.path.exists(audio_file):
            print(f"错误：音频文件不存在: {audio_file}")
            sys.exit(1)
            
        features = extract_features(audio_file)
        print("特征维度:", features.shape)
        print("特征值:", features)
        
        print(f"程序总执行时间: {time.time() - total_start_time:.2f}秒")
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        sys.exit(1)