import streamlit as st
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import librosa
import librosa.display
import tempfile
import shutil
from datetime import datetime
from utils.audio_separator import get_available_gpus, get_recommended_threads

matplotlib.use('Agg')

# 设置matplotlib样式
plt.style.use('default')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 将项目根目录添加到 sys.path
project_root = os.path.dirname(os.path.abspath(__file__))



sys.path.append(project_root)









# --- 配置 ---
SUPPORTED_EXTENSIONS = ["mp3", "wav", "flac", "ogg", "m4a"]
DEFAULT_MODEL = "htdemucs"

AVAILABLE_MODELS = ["htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx", "mdx_extra", "mdx_q", "mdx_u", "hdemucs_mmi"]


OUTPUT_BASE_DIR = os.path.join(project_root, "demucs_separated_output")
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# 绘图函数
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




    img = librosa.display.specshow(mel_spect_db, y_axis='mel', x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('梅尔频谱图')
    plt.tight_layout()
    return fig


def process_audio_file(uploaded_file, params, status_placeholder, results_placeholder):
    """处理单个音频文件"""
    status_placeholder.info("⏳ 正在处理音频，请稍候...")
    
    # 创建唯一的运行目录
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_output_dir = os.path.join(OUTPUT_BASE_DIR, run_id)
    os.makedirs(run_output_dir, exist_ok=True)

    # 保存上传的文件到临时位置
    temp_dir = tempfile.mkdtemp(dir=run_output_dir)
    temp_input_path = os.path.join(temp_dir, uploaded_file.name)
    
    try:
        with open(temp_input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # 加载音频用于可视化
        y, sr = librosa.load(temp_input_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # 调用分离函数
        from utils.audio_separator import separate_audio
        start_time = datetime.now()
        
        success = separate_audio(
            track_path=temp_input_path,
            output_dir=os.path.join(run_output_dir, "separated"),
            model_name=params["model_name"],
            device=params["device"],
            shifts=params["shifts"],
            overlap=params["overlap"],
            no_split=params["no_split"],
            segment=params["segment"],
            two_stems=params["two_stems"],
            clip_mode=params["clip_mode"],
            mp3_bitrate=params["mp3_bitrate"],
            mp3_preset=params["mp3_preset"],
            filename=params["filename"],
            jobs=params["jobs"],
            verbose=True
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        if success:
            display_results(uploaded_file, run_output_dir, params, y, sr, status_placeholder, results_placeholder, processing_time)
        else:
            status_placeholder.error("❌ 音频分离失败")
            
    except Exception as e:
        status_placeholder.error(f"处理过程中发生错误: {e}")
        import traceback
        st.error(traceback.format_exc())
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def display_results(uploaded_file, run_output_dir, params, y, sr, status_placeholder, results_placeholder, processing_time):
    """显示处理结果"""
    status_placeholder.success(f"✅ 分离完成！耗时: {processing_time:.2f} 秒")
    
    # 查找输出文件
    output_subdir = os.path.join(run_output_dir, "separated", params["model_name"])
    original_name = os.path.splitext(uploaded_file.name)[0]
    
    # 根据分离模式确定要查找的词干
    stems = [params["two_stems"], f"no_{params['two_stems']}"] if params["two_stems"] else ["vocals", "drums", "bass", "other"]
    results = {}
    
    for stem in stems:
        pattern = f"{original_name}_{stem}.*"
        for f in os.listdir(output_subdir):
            if f.startswith(f"{original_name}_{stem}"):
                results[stem] = os.path.join(output_subdir, f)
                break
    
    with results_placeholder:
        # 显示音频分析图
        if y is not None and sr is not None:
            st.subheader("原始音频分析")
            col1, col2 = st.columns(2)
            with col1:
                try:
                    st.pyplot(plot_waveform(y, sr))
                except Exception as e:
                    st.error(f"波形图生成失败: {e}")
            with col2:
                try:
                    st.pyplot(plot_melspectrogram(y, sr))
                except Exception as e:
                    st.error(f"梅尔频谱图生成失败: {e}")
        
        # 显示分离结果
        st.subheader("分离的音轨")
        if params["two_stems"]:
            cols = st.columns(2)
            for i, (stem, path) in enumerate(results.items()):
                with cols[i % 2]:
                    display_stem(stem, path, params["mp3_output"])
        else:
            tabs = st.tabs(["人声", "鼓声", "贝斯", "其他"])
            stem_map = {"vocals": 0, "drums": 1, "bass": 2, "other": 3}
            for stem, path in results.items():
                with tabs[stem_map.get(stem, 0)]:
                    display_stem(stem, path, params["mp3_output"])

def display_stem(stem, path, is_mp3):
    """显示单个音轨"""
    stem_name = {
        "vocals": "人声", "no_vocals": "伴奏",
        "drums": "鼓声", "bass": "贝斯", "other": "其他"
    }.get(stem, stem)
    
    st.markdown(f"**{stem_name}**")
    st.audio(path)
    
    ext = ".mp3" if is_mp3 else ".wav"
    with open(path, "rb") as f:
        st.download_button(
            label=f"下载 {stem_name}{ext}",
            data=f,
            file_name=os.path.basename(path),
            mime=f"audio/{ext.lstrip('.')}"
        )

def main():
    st.set_page_config(layout="wide", page_title="音频分离与降噪 (Demucs)")

    st.title("🎵 音频分离与降噪 (Demucs)")

    
    # 模式选择
    mode = st.radio("处理模式", ["单文件处理", "批量处理"], horizontal=True)
    
    if mode == "单文件处理":
        process_single_file_ui()
    else:
        process_batch_ui()


def process_single_file_ui():
    """单文件处理界面"""
    col1, col2 = st.columns(2)

    
    with col1:




































        st.subheader("上传与参数")
        uploaded_file = st.file_uploader("选择音频文件", type=SUPPORTED_EXTENSIONS)
        params = get_separation_params()
        















    with col2:
        st.subheader("分离结果")
        status_placeholder = st.empty()
        results_placeholder = st.container()
    
    if st.button("🚀 开始分离", disabled=not uploaded_file):
        if uploaded_file:
            process_audio_file(uploaded_file, params, status_placeholder, results_placeholder)
        else:
            status_placeholder.warning("请先上传音频文件")



def process_batch_ui():
    """批量处理界面"""
    st.warning("批量处理功能仍在开发中，当前版本仅支持单文件处理")
    # 这里可以添加批量处理UI的实现
























































































































































































def get_separation_params():
    """获取分离参数"""
    params = {}
    
    with st.expander("基本参数"):
        params["model_name"] = st.selectbox(
            "模型", AVAILABLE_MODELS, 
            index=AVAILABLE_MODELS.index(DEFAULT_MODEL)
        )
        




        # 设备选择
        is_cuda = torch.cuda.is_available()
        params["device"] = st.radio(
            "设备", ["cuda", "cpu"], 
            index=0 if is_cuda else 1,
            horizontal=True
        )
        

        # 分离模式
        separation_mode = st.radio(
            "分离模式", ["标准四轨分离", "仅人声/伴奏分离"],
            index=1, horizontal=True
        )
        params["two_stems"] = "vocals" if separation_mode == "仅人声/伴奏分离" else None
    
    with st.expander("高级参数"):
        col1, col2 = st.columns(2)
        







        with col1:
            params["shifts"] = st.slider("随机移位", 0, 20, 2)
            params["overlap"] = st.slider("重叠比例", 0.0, 1.0, 0.5, 0.05)
            params["segment"] = st.slider("分段大小(秒)", 1, 30, 7)
            params["jobs"] = st.slider("并行数", 1, 16, get_recommended_threads())
            
        with col2:
            params["no_split"] = st.checkbox("禁用分段处理", False)
            params["clip_mode"] = st.selectbox("削波处理", ["rescale", "clamp"], 0)
            
            # 输出格式
            params["mp3_output"] = st.checkbox("输出为MP3", False)
            if params["mp3_output"]:
                params["mp3_bitrate"] = st.slider("MP3比特率", 64, 320, 256, 32)
                params["mp3_preset"] = st.select_slider("MP3质量", options=[2,3,4,5,6,7], value=2)
            else:
                params["mp3_bitrate"] = 320
                params["mp3_preset"] = 2
                
            params["filename"] = "{track}_{stem}.{ext}"
    
    return params

if __name__ == "__main__":
    main()