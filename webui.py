import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import os
from utils.feature_extract import extract_features
from events_guess import predict_audio_events
import tempfile
import matplotlib
import io
matplotlib.use('Agg')  # 必须在导入 pyplot 之前设置后端

# 设置matplotlib样式
plt.style.use('default')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_waveform(y, sr):
    """绘制音频波形图"""
    fig = plt.figure(figsize=(12, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.linspace(0, len(y)/sr, len(y)), y)
    ax.set_title('音频波形图')
    ax.set_xlabel('时间 (秒)')
    ax.set_ylabel('振幅')
    ax.grid(True)
    plt.tight_layout()
    return fig

def plot_melspectrogram(y, sr):
    """绘制梅尔频谱图"""
    fig = plt.figure(figsize=(12, 3))
    ax = fig.add_subplot(1, 1, 1)
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
    img = librosa.display.specshow(mel_spect_db, 
                                 y_axis='mel', 
                                 x_axis='time', 
                                 ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('梅尔频谱图')
    plt.tight_layout()
    return fig

def plot_mfcc(y, sr):
    """绘制MFCC特征图"""
    fig = plt.figure(figsize=(12, 3))
    ax = fig.add_subplot(1, 1, 1)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    img = librosa.display.specshow(mfccs, 
                                 x_axis='time', 
                                 ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set_title('MFCC特征图')
    plt.tight_layout()
    return fig

def main():
    # 设置页面配置
    st.set_page_config(
        page_title="音频事件分析系统",
        layout="wide"
    )

    # 添加侧边栏
    st.sidebar.title("参数设置")
    window_size = st.sidebar.slider(
        "窗口大小 (秒)",
        min_value=0.5,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="音频分析的时间窗口大小"
    )
    
    hop_length = st.sidebar.slider(
        "滑动步长 (秒)",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
        help="连续窗口之间的时间间隔"
    )
    
    confidence_threshold = st.sidebar.slider(
        "置信度阈值",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="事件检测的置信度阈值"
    )

    st.title("音频事件分析系统")
    st.write("上传音频文件进行分析")

    # 文件上传
    uploaded_file = st.file_uploader("选择音频文件", type=['wav', 'mp3'])

    if uploaded_file is not None:
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # 直接在内存中处理上传的文件
            audio_bytes = uploaded_file.read()
            audio_buffer = io.BytesIO(audio_bytes)
            
            status_text.text("正在加载音频...")
            progress_bar.progress(20)

            # 从内存缓冲区加载音频文件
            y, sr = librosa.load(audio_buffer, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # 重置缓冲区位置
            audio_buffer.seek(0)
            
            status_text.text("正在进行事件检测...")
            progress_bar.progress(60)
            
            # 直接传递内存缓冲区进行事件检测
            events = predict_audio_events(
                audio_buffer,
                window_size=window_size,
                hop_length=hop_length,
                confidence_threshold=confidence_threshold
            )

            # 使用列布局优化显示效果
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # 显示波形图
                st.subheader("波形图")
                try:
                    plt.close('all')  # 清除所有图形
                    fig1 = plot_waveform(y, sr)
                    st.pyplot(fig1)
                    plt.close(fig1)
                except Exception as e:
                    st.error(f"波形图生成失败: {str(e)}")

                # 显示梅尔频谱图
                st.subheader("梅尔频谱图")
                try:
                    plt.close('all')  # 清除所有图形
                    fig2 = plot_melspectrogram(y, sr)
                    st.pyplot(fig2)
                    plt.close(fig2)
                except Exception as e:
                    st.error(f"梅尔频谱图生成失败: {str(e)}")

                # 显示MFCC特征图
                st.subheader("MFCC特征图")
                try:
                    plt.close('all')  # 清除所有图形
                    fig3 = plot_mfcc(y, sr)
                    st.pyplot(fig3)
                    plt.close(fig3)
                except Exception as e:
                    st.error(f"MFCC特征图生成失败: {str(e)}")

            with col2:
                # 显示音频信息
                st.subheader("音频信息")
                st.write(f"采样率: {sr} Hz")
                st.write(f"时长: {duration:.2f} 秒")

                # 显示检测结果
                st.subheader("事件检测结果")
                if events:
                    for event, start, end, confidence in events:
                        with st.expander(f"事件: {event} ({confidence:.2%})"):
                            st.write(f"开始时间: {start:.2f}秒")
                            st.write(f"结束时间: {end:.2f}秒")
                else:
                    st.write("未检测到显著事件")

            status_text.text("分析完成！")
            progress_bar.progress(100)

        except Exception as e:
            st.error(f"处理过程中出错: {str(e)}")
        finally:
            audio_buffer.close()

if __name__ == "__main__":
    main()
