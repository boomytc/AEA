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
    # 参数验证
    if not params["no_split"] and params["segment"] > 7.8:
        status_placeholder.error("❌ 分段大小不能超过7.8秒")
        return
        
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
    process_single_file_ui()  # 直接调用单文件处理界面

def process_single_file_ui():
    """单文件处理界面"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("上传与参数")
        uploaded_file = st.file_uploader("选择音频文件", type=SUPPORTED_EXTENSIONS)
        
        # 添加音频预览功能
        if uploaded_file is not None:
            st.subheader("原始音频预览")
            st.audio(uploaded_file)
            
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

def get_separation_params():
    """获取分离参数"""
    params = {}
    
    # 修改为默认展开的基本参数
    with st.expander("基本参数", expanded=True):  # 添加 expanded=True
        # 固定使用htdemucs模型
        params["model_name"] = "htdemucs"
        
        # 设备选择
        is_cuda = torch.cuda.is_available()
        params["device"] = st.radio(
            "设备", ["cuda", "cpu"], 
            index=0 if is_cuda else 1,
            horizontal=True
        )
        
        # 固定使用人声/伴奏分离模式
        params["two_stems"] = "vocals"
    
    # 修改为默认展开的高级参数
    with st.expander("高级参数", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            params["shifts"] = st.slider(
                "随机移位", 0, 20, 2,
                help="增加随机移位次数可以提高分离质量但会延长处理时间。0表示禁用，推荐值2-5"
            )
            # 修改overlap参数，限制最大值为0.99
            params["overlap"] = st.slider(
                "重叠比例", 0.0, 0.99, 0.5, 0.05,
                help="音频片段之间的重叠比例(0-0.99)。增加可以提高质量但会使用更多内存"
            )
            
            # 修改segment参数为整数类型
            max_segment = 7  # 改为整数7
            params["segment"] = st.slider(
                "分段大小(秒)", 1, max_segment, min(7, max_segment), 1,
                help=f"将音频分割成小段处理，有助于节省显存。最大不能超过{max_segment}秒"
            )
            
            params["jobs"] = st.slider(
                "并行数", 1, 16, get_recommended_threads(),
                help="并行处理线程数。自动设置为CPU核心数-2，增加可加快处理但会占用更多资源"
            )
            
        with col2:
            # 禁用分段处理选项需要与segment参数联动
            params["no_split"] = st.checkbox(
                "禁用分段处理", False,
                help="禁用音频分段处理，可能提高质量但会显著增加显存占用"
            )
            
            # 如果禁用分段处理，则强制使用CPU
            if params["no_split"]:
                st.warning("⚠️ 禁用分段处理将使用CPU处理以避免内存问题")
                params["device"] = "cpu"  # 强制使用CPU
                params["segment"] = None  # 设置为None表示不分割
            
            params["clip_mode"] = st.selectbox(
                "削波处理", ["rescale", "clamp"], 0,
                help="避免削波的策略：rescale(动态缩放整个信号)或clamp(直接限制振幅)"
            )
            
            # 输出格式
            params["mp3_output"] = st.checkbox(
                "输出为MP3", False,
                help="将输出转换为MP3格式以减小文件大小"
            )
            if params["mp3_output"]:
                params["mp3_bitrate"] = st.slider(
                    "MP3比特率", 64, 320, 256, 32,
                    help="MP3编码比特率(kbps)，值越高音质越好但文件越大"
                )
                params["mp3_preset"] = st.select_slider(
                    "MP3质量", options=[2,3,4,5,6,7], value=2,
                    help="MP3编码质量预设：2(最高质量)到7(最快速度)"
                )
            else:
                params["mp3_bitrate"] = 320
                params["mp3_preset"] = 2
                
            params["filename"] = "{track}_{stem}.{ext}"
    
    return params

if __name__ == "__main__":
    main()
    # Add this right after torch import to prevent Streamlit inspection issues
    torch.classes.__path__ = None  # Disable Streamlit's attempt to inspect torch.classes