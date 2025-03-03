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

if 'has_displayed_results' not in st.session_state:
    st.session_state.has_displayed_results = False

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
    
    # 添加模型选择
    model_type = st.sidebar.radio(
        "选择模型类型",
        ["随机森林 (RF)", "XGBoost (XGB)"],
        index=1,  # 默认选择XGBoost
        help="选择用于检测的模型类型"
    )
    
    # 转换为代码中使用的模型类型
    model_code_type = "xgb" if "XGBoost" in model_type else "rf"
    
    # 高级选项折叠面板
    with st.sidebar.expander("高级选项"):
        custom_model_path = st.text_input(
            "自定义模型路径",
            value="",
            help="如果要使用自定义模型，请输入完整路径"
        )
    
    window_size = st.sidebar.slider(
        "窗口大小 (秒)",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.1,
        help="音频分析的时间窗口大小"
    )
    
    hop_length = st.sidebar.slider(
        "滑动步长 (秒)",
        min_value=0.1,
        max_value=2.0,
        value=1.5,
        step=0.1,
        help="连续窗口之间的时间间隔"
    )
    
    confidence_threshold = st.sidebar.slider(
        "置信度阈值",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help="事件检测的置信度阈值"
    )

    st.title("音频事件分析系统")
    st.write("上传音频文件进行分析")
    
    # 显示当前使用的模型
    st.markdown(f"**当前模型**: {model_type}")

    # 文件上传
    uploaded_file = st.file_uploader("选择音频文件", type=['wav', 'mp3'])

    if uploaded_file is not None:
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        result_placeholder = st.empty()

        # 如果已经显示过结果，显示保存的图片和信息
        if st.session_state.has_displayed_results:
            with result_placeholder.container():
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if 'waveform' in st.session_state:
                        st.subheader("波形图")
                        st.pyplot(st.session_state.waveform)
                    
                    if 'melspectrogram' in st.session_state:
                        st.subheader("梅尔频谱图")
                        st.pyplot(st.session_state.melspectrogram)
                    
                    if 'mfcc' in st.session_state:
                        st.subheader("MFCC特征图")
                        st.pyplot(st.session_state.mfcc)
                
                with col2:
                    if 'audio_info' in st.session_state:
                        st.subheader("音频信息")
                        st.write(st.session_state.audio_info)
                    
                    if 'events' in st.session_state:
                        st.subheader("事件检测结果")
                        if st.session_state.events:
                            for event, start, end, confidence in st.session_state.events:
                                with st.expander(f"事件: {event} ({confidence:.2%})"):
                                    st.write(f"开始时间: {start:.2f}秒")
                                    st.write(f"结束时间: {end:.2f}秒")
                        else:
                            st.write("未检测到显著事件")

        if st.button("开始检测"):
            st.session_state.has_displayed_results = True
            try:
                # 清空之前的结果
                result_placeholder.empty()
                
                # 显示进度条和状态文本
                progress_bar = progress_placeholder.progress(0)
                status_text = status_placeholder.text("正在加载音频...")

                # 创建临时文件
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                    # 保存上传的文件到临时文件
                    temp_audio.write(uploaded_file.read())
                    temp_audio_path = temp_audio.name

                progress_bar.progress(20)

                # 从临时文件加载音频
                y, sr = librosa.load(temp_audio_path, sr=None)
                duration = librosa.get_duration(y=y, sr=sr)
                
                status_text.text(f"正在使用{model_type}模型进行事件检测...")
                progress_bar.progress(60)
                
                # 准备模型参数
                model_params = {
                    "window_size": window_size,
                    "hop_length": hop_length,
                    "confidence_threshold": confidence_threshold,
                    "model_type": model_code_type
                }
                
                # 根据模型类型设置默认模型路径
                if not custom_model_path:
                    if model_code_type == 'rf':
                        model_params["model_path"] = "models/audio_event_model_segments.pkl"
                    else:  # xgb
                        model_params["model_path"] = "models/audio_event_model_xgboost.pkl"
                else:
                    # 如果有自定义模型路径
                    model_params["model_path"] = custom_model_path
                
                # 使用临时文件路径进行事件检测
                try:
                    events = predict_audio_events(
                        temp_audio_path,
                        **model_params
                    )
                except Exception as e:
                    st.error(f"事件检测失败: {str(e)}")
                    import traceback
                    st.error(f"详细错误: {traceback.format_exc()}")
                    events = []

                # 删除临时文件
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass

                # 使用列布局显示结果
                with result_placeholder.container() as results:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # 显示波形图
                        st.subheader("波形图")
                        try:
                            plt.close('all')
                            fig1 = plot_waveform(y, sr)
                            st.pyplot(fig1)
                            st.session_state.waveform = fig1
                        except Exception as e:
                            st.error(f"波形图生成失败: {str(e)}")

                        # 显示梅尔频谱图
                        st.subheader("梅尔频谱图")
                        try:
                            plt.close('all')
                            fig2 = plot_melspectrogram(y, sr)
                            st.pyplot(fig2)
                            st.session_state.melspectrogram = fig2
                        except Exception as e:
                            st.error(f"梅尔频谱图生成失败: {str(e)}")

                        # 显示MFCC特征图
                        st.subheader("MFCC特征图")
                        try:
                            plt.close('all')
                            fig3 = plot_mfcc(y, sr)
                            st.pyplot(fig3)
                            st.session_state.mfcc = fig3
                        except Exception as e:
                            st.error(f"MFCC特征图生成失败: {str(e)}")

                    with col2:
                        # 显示音频信息
                        st.subheader("音频信息")
                        audio_info = {
                            "采样率": f"{sr} Hz",
                            "时长": f"{duration:.2f} 秒",
                            "使用模型": model_type
                        }
                        st.write(audio_info)
                        st.session_state.audio_info = audio_info

                        # 显示检测结果
                        st.subheader("事件检测结果")
                        st.session_state.events = events
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
                status_placeholder.error(f"处理过程中出错: {str(e)}")

if __name__ == "__main__":
    main()
