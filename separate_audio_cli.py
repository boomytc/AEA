import argparse
import os
import shutil
import sys
from utils.audio_separator import separate_audio

def move_files_from_model_dir(track_path, output_dir, model_name, save_file=None):
    """
    将文件从模型子目录移动到目标目录，并删除原始文件夹及其文件
    
    参数:
        track_path: 音频文件路径
        output_dir: 输出目录
        model_name: 模型名称
        save_file: 指定保存的声部（'vocals' 或 'no_vocals'），如果为 None 则保存所有声部
    
    返回:
        bool: 是否成功移动文件
    """
    model_dir = os.path.join(output_dir, model_name)
    if not os.path.exists(model_dir):
        print(f"模型目录不存在: {model_dir}")
        return False
    
    # 获取音频文件名（不包含路径和扩展名）
    track_name = os.path.splitext(os.path.basename(track_path))[0]
    file_ext = os.path.splitext(track_path)[1]  # 获取原始文件扩展名
    if not file_ext:  # 如果原文件没有扩展名，默认使用.wav
        file_ext = ".wav"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 复制指定声部的文件到目标目录，并重命名
    copied_files = []
    source_files = []
    
    # 定义文件匹配和重命名规则
    file_mapping = {
        "vocals": {"pattern": f"{track_name}_vocals", "new_name": f"{track_name}_voice"},
        "no_vocals": {"pattern": f"{track_name}_no_vocals", "new_name": f"{track_name}_ambient"}
    }
    
    # 遍历模型目录中的文件
    for file in os.listdir(model_dir):
        if file.startswith(track_name):
            # 确定文件类型（vocals 或 no_vocals）
            file_type = None
            new_filename = None
            
            # 判断文件类型并确定新文件名
            for stem_type, mapping in file_mapping.items():
                if mapping["pattern"] in file:
                    file_type = stem_type
                    # 保留原始文件扩展名
                    file_ext_current = os.path.splitext(file)[1]
                    new_filename = f"{mapping['new_name']}{file_ext_current}"
                    break
            
            # 如果指定了声部且当前文件不匹配，则跳过
            if save_file and file_type != save_file:
                print(f"  跳过非指定声部文件: {file}")
                continue
            
            # 如果无法确定文件类型，使用原始文件名
            if not new_filename:
                new_filename = file
            
            src_path = os.path.join(model_dir, file)
            dst_path = os.path.join(output_dir, new_filename)
            
            # 复制并重命名文件
            shutil.copy2(src_path, dst_path)  # 复制文件，保留元数据
            copied_files.append(dst_path)
            source_files.append(src_path)
    
    if copied_files:
        if save_file:
            print(f"已将指定声部({save_file})文件复制到: {output_dir}")
        else:
            print(f"已将所有声部文件复制到: {output_dir}")
        
        for file in copied_files:
            print(f"  - {os.path.basename(file)}")
        
        # 删除原始文件
        for file in source_files:
            try:
                os.remove(file)
                print(f"  删除原始文件: {os.path.basename(file)}")
            except Exception as e:
                print(f"  删除原始文件失败: {os.path.basename(file)}, 错误: {e}")
        
        # 删除原始目录，不管其中是否还有其他文件
        try:
            # 首先删除目录中的所有文件
            remaining_files = os.listdir(model_dir)
            if remaining_files:
                print(f"  删除模型目录中的其他文件: {len(remaining_files)} 个文件")
                for file in remaining_files:
                    try:
                        file_path = os.path.join(model_dir, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            print(f"    - 删除文件: {file}")
                        elif os.path.isdir(file_path):
                            # 如果有子目录，使用shutil.rmtree递归删除
                            shutil.rmtree(file_path)
                            print(f"    - 删除子目录: {file}")
                    except Exception as e:
                        print(f"    - 删除文件失败: {file}, 错误: {e}")
            
            # 删除模型目录
            os.rmdir(model_dir)
            print(f"  已删除模型目录: {model_dir}")
        except Exception as e:
            print(f"  删除目录失败: {model_dir}, 错误: {e}")
        
        return True
    else:
        print(f"未找到要移动的文件在: {model_dir}")
        return False

def select_device(requested_device):
    if requested_device and requested_device != "auto":
        return requested_device

    try:
        import torch
    except Exception:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def main():
    parser = argparse.ArgumentParser(description="音频分离脚本 (Demucs)")
    parser.add_argument("input", help="音频文件路径")
    parser.add_argument("-o", "--output", default="test_separated", help="输出目录")
    parser.add_argument("-n", "--model", default="htdemucs", help="预训练模型名称")
    parser.add_argument(
        "-d",
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="使用的设备 (auto 将自动选择)"
    )
    parser.add_argument(
        "--save-file",
        choices=["vocals", "no_vocals", "all"],
        default="all",
        help="仅保留指定声部文件"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="显示详细输出")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"错误：音频文件不存在: {args.input}")
        sys.exit(1)

    device = select_device(args.device)
    save_file = None if args.save_file == "all" else args.save_file

    separate_result = separate_audio(
        args.input,
        output_dir=args.output,
        model_name=args.model,
        device=device,
        two_stems="vocals",  # 仅分离人声和伴奏
        verbose=args.verbose,
        filename="{track}_{stem}.{ext}"  # 设置输出文件名格式
    )

    if not separate_result:
        print("音频分离失败")
        sys.exit(1)

    move_result = move_files_from_model_dir(args.input, args.output, args.model, save_file)
    if move_result:
        print(f"最终结果保存在: {args.output}")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
