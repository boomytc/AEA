import argparse
import io
import os
import sys
import tempfile
import time
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

import joblib
import librosa
import numpy as np
import soundfile as sf
import xgboost as xgb

from utils.audio_separator import separate_audio
from utils.feature_extract import extract_features

DEFAULT_RF_MODEL_PATH = "models/audio_event_model_segments.pkl"
DEFAULT_XGB_MODEL_PATH = "models/audio_event_model_xgboost.pkl"
DEFAULT_SCALER_PATH = "models/feature_scaler_segments.pkl"

def resolve_model_paths(model_type, model_path, scaler_path):
    if not model_path:
        model_path = DEFAULT_XGB_MODEL_PATH if model_type == "xgb" else DEFAULT_RF_MODEL_PATH
    if not scaler_path:
        scaler_path = DEFAULT_SCALER_PATH
    return model_path, scaler_path

def load_model_and_scaler(model_type, model_path, scaler_path):
    model_path, scaler_path = resolve_model_paths(model_type, model_path, scaler_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}，请先训练模型或检查路径")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"标准化器文件不存在: {scaler_path}，请先训练模型或检查路径")

    model_data = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    if model_type == "xgb":
        if not isinstance(model_data, dict) or "model" not in model_data or "label_mapping" not in model_data:
            raise ValueError("XGBoost模型文件格式不正确或模型类型不匹配")
    elif model_type == "rf":
        if not hasattr(model_data, "predict") or not hasattr(model_data, "predict_proba"):
            raise ValueError("随机森林模型文件格式不正确或模型类型不匹配")
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    if not hasattr(scaler, "transform"):
        raise ValueError("标准化器文件格式不正确或未加载成功")

    return model_data, scaler

def select_separation_device():
    try:
        import torch
    except Exception:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def analyze_audio_segment(segment, sr, model_data, scaler, segment_start_time, model_type="rf"):
    """
    分析音频片段并返回预测结果
    """
    buffer = io.BytesIO()
    try:
        sf.write(buffer, segment, sr, format="WAV")
        buffer.seek(0)

        features = extract_features(buffer)
        features = features.reshape(1, -1)
        features_scaled = scaler.transform(features)

        if model_type == "rf":
            prediction = model_data.predict(features_scaled)[0]
            probabilities = model_data.predict_proba(features_scaled)[0]
            max_prob = np.max(probabilities)
        elif model_type == "xgb":
            model = model_data["model"]
            label_mapping = model_data["label_mapping"]
            dmatrix = xgb.DMatrix(features_scaled)
            pred_probs = model.predict(dmatrix)
            pred_idx = np.argmax(pred_probs, axis=1)[0]
            prediction = str(label_mapping[int(pred_idx)])
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
        if event[0] == current_event[0]:
            current_confidence.append(event[3])
        else:
            merged.append((
                current_event[0],
                current_start,
                event[1],
                np.mean(current_confidence)
            ))
            current_event = event
            current_start = event[1]
            current_confidence = [event[3]]

    merged.append((
        current_event[0],
        current_start,
        current_event[2],
        np.mean(current_confidence)
    ))

    return merged

def move_files_from_model_dir(track_path, output_dir, model_name, save_file=None):
    model_dir = os.path.join(output_dir, model_name)
    if not os.path.exists(model_dir):
        print(f"模型目录不存在: {model_dir}")
        return None

    track_name = os.path.splitext(os.path.basename(track_path))[0]
    file_ext = os.path.splitext(track_path)[1]
    if not file_ext:
        file_ext = ".wav"

    os.makedirs(output_dir, exist_ok=True)
    no_vocals_path = None

    file_mapping = {
        "vocals": {"pattern": f"{track_name}_vocals", "new_name": f"{track_name}_voice"},
        "no_vocals": {"pattern": f"{track_name}_no_vocals", "new_name": f"{track_name}_ambient"}
    }

    for file in os.listdir(model_dir):
        if file.startswith(track_name):
            file_type = None
            new_filename = None

            for stem_type, mapping in file_mapping.items():
                if mapping["pattern"] in file:
                    file_type = stem_type
                    file_ext_current = os.path.splitext(file)[1]
                    new_filename = f"{mapping['new_name']}{file_ext_current}"
                    break

            if save_file and file_type != save_file:
                print(f"  跳过非指定声部文件: {file}")
                continue

            if not new_filename:
                new_filename = file

            src_path = os.path.join(model_dir, file)
            dst_path = os.path.join(output_dir, new_filename)
            shutil.copy2(src_path, dst_path)
            print(f"  - 已复制文件: {os.path.basename(dst_path)}")

            if file_type == "no_vocals" or (save_file == "no_vocals" and file_type == save_file):
                no_vocals_path = dst_path

    try:
        shutil.rmtree(model_dir)
        print(f"  已删除模型目录: {model_dir}")
    except Exception as e:
        print(f"  删除目录失败: {model_dir}, 错误: {e}")

    return no_vocals_path

def separate_and_get_ambient(audio_file, temp_dir=None, device=None):
    print(f"开始分离音频文件: {audio_file}")

    created_temp_dir = False
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="audio_separate_")
        created_temp_dir = True
    else:
        os.makedirs(temp_dir, exist_ok=True)

    try:
        model_name = "htdemucs"
        if device is None:
            device = select_separation_device()
        separate_result = separate_audio(
            audio_file,
            output_dir=temp_dir,
            model_name=model_name,
            device=device,
            two_stems="vocals",
            verbose=True,
            filename="{track}_{stem}.{ext}"
        )

        if not separate_result:
            print(f"音频分离失败，将使用原始音频文件进行事件检测: {audio_file}")
            return audio_file

        ambient_file = move_files_from_model_dir(audio_file, temp_dir, model_name, save_file="no_vocals")

        if ambient_file:
            print(f"成功提取无人声音频: {ambient_file}")
            return ambient_file
        print(f"提取无人声音频失败，将使用原始音频文件进行事件检测: {audio_file}")
        return audio_file
    except Exception as e:
        print(f"音频分离过程中出错: {str(e)}")
        print(f"将使用原始音频文件进行事件检测: {audio_file}")
        return audio_file
    finally:
        if created_temp_dir and not os.path.exists(os.path.join(temp_dir, os.path.basename(audio_file).replace(".", "_ambient."))):
            try:
                if not os.listdir(temp_dir):
                    shutil.rmtree(temp_dir)
                    print(f"已删除临时目录: {temp_dir}")
            except Exception as e:
                print(f"删除临时目录失败: {temp_dir}, 错误: {e}")

def predict_audio_events(
    audio_file: str,
    window_size: float = 2.0,
    hop_length: float = 1.0,
    confidence_threshold: float = 0.55,
    model_path: str = None,
    scaler_path: str = None,
    model_type: str = "rf",
    use_ambient_only: bool = False
) -> List[Tuple[str, float, float, float]]:
    """
    对音频文件进行多事件检测
    """
    process_start_time = time.time()

    if window_size <= 0:
        raise ValueError("窗口大小必须大于0")
    if hop_length <= 0:
        raise ValueError("滑动步长必须大于0")
    if confidence_threshold < 0 or confidence_threshold > 1:
        raise ValueError("置信度阈值必须在0到1之间")

    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"音频文件不存在: {audio_file}")

    model_data, scaler = load_model_and_scaler(model_type, model_path, scaler_path)

    ambient_file = audio_file
    if use_ambient_only:
        print("启用了无人声模式，将先进行音频分离...")
        temp_dir = os.path.join(os.path.dirname(audio_file), "temp_separated_" + os.path.splitext(os.path.basename(audio_file))[0])
        ambient_file = separate_and_get_ambient(audio_file, temp_dir)

    y, sr = librosa.load(ambient_file, sr=None)
    if len(y) == 0:
        raise ValueError("音频数据为空或无法读取")

    duration = librosa.get_duration(y=y, sr=sr)
    print(f"开始分析音频文件: {ambient_file}")
    print(f"音频长度: {duration:.2f}秒")
    print(f"使用模型类型: {model_type}")

    window_samples = int(window_size * sr)
    hop_samples = int(hop_length * sr)
    if window_samples <= 0 or hop_samples <= 0:
        raise ValueError("窗口大小或滑动步长过小，导致采样点数为0")
    total_samples = len(y)

    with ThreadPoolExecutor() as executor:
        futures = []
        if total_samples < window_samples:
            start_samples = [0]
        else:
            start_samples = range(0, total_samples - window_samples + 1, hop_samples)

        for start_sample in start_samples:
            segment = y[start_sample:start_sample + window_samples]
            if len(segment) == 0:
                continue
            segment_start_time = start_sample / sr
            segment_end_time = (start_sample + len(segment)) / sr
            future = executor.submit(analyze_audio_segment, segment, sr, model_data, scaler, segment_start_time, model_type)
            futures.append((segment_start_time, segment_end_time, future))

        window_predictions = []
        for segment_start_time, segment_end_time, future in futures:
            try:
                prediction, confidence = future.result()
                print(f"时间窗口 {segment_start_time:.1f}s - {segment_end_time:.1f}s:")
                print(f"  预测事件: {prediction}")
                print(f"  置信度: {confidence:.2%}")
                if confidence >= confidence_threshold:
                    window_predictions.append((prediction, segment_start_time, segment_end_time, confidence))
            except Exception as e:
                print(f"处理时间窗口 {segment_start_time:.1f}s 时出错: {str(e)}")

    merged_predictions = merge_predictions(window_predictions)

    print("\n检测到的事件:")
    for event, start, end, confidence in merged_predictions:
        print(f"事件: {event}, 开始时间: {start:.2f}s, 结束时间: {end:.2f}s, 置信度: {confidence:.2%}")

    process_end_time = time.time()
    print(f"\n总执行时间: {process_end_time - process_start_time:.2f}秒")

    return merged_predictions

def main():
    parser = argparse.ArgumentParser(description="音频事件检测 CLI")
    parser.add_argument("audio_file", help="音频文件路径")
    parser.add_argument("--model_type", "-m", choices=["rf", "xgb"], default="rf", help="模型类型：rf (随机森林) 或 xgb (XGBoost)")
    parser.add_argument("--model_path", help="模型文件路径")
    parser.add_argument("--scaler_path", help="标准化器文件路径")
    parser.add_argument("--window_size", "-w", type=float, default=2.0, help="分析窗口大小（秒）")
    parser.add_argument("--hop_length", "-l", type=float, default=1.0, help="窗口滑动步长（秒）")
    parser.add_argument("--confidence", "-c", type=float, default=0.55, help="置信度阈值")
    parser.add_argument("--ambient-only", action="store_true", help="启用无人声模式（先分离音频，仅用无人声部分检测）")

    args = parser.parse_args()

    model_path = args.model_path
    if model_path is None:
        model_path = DEFAULT_XGB_MODEL_PATH if args.model_type == "xgb" else DEFAULT_RF_MODEL_PATH

    try:
        predict_audio_events(
            args.audio_file,
            window_size=args.window_size,
            hop_length=args.hop_length,
            confidence_threshold=args.confidence,
            model_path=model_path,
            scaler_path=args.scaler_path,
            model_type=args.model_type,
            use_ambient_only=args.ambient_only
        )
    except Exception as e:
        print(f"检测失败: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
