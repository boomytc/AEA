import os
import subprocess
import argparse
import time

def separate_audio(
    track_path,
    output_dir="separated",
    model_name="htdemucs",
    device="cuda",
    shifts=8,
    overlap=0.5,
    no_split=False,
    segment=7,
    two_stems="vocals",
    clip_mode="rescale",
    mp3_bitrate=256,
    mp3_preset=2,
    filename=None,
    jobs=0,  # 修改默认值为0，让demucs自动选择最佳并行数
    verbose=False
):
    """
    使用 Demucs 进行音频分离
    
    参数:
        track_path: 音频文件路径
        output_dir: 输出目录
        model_name: 预训练模型名称
        device: 使用的设备 (cuda 或 cpu)
        shifts: 随机移位数量 (增加质量但需要更多时间)
        overlap: 分割之间的重叠
        no_split: 不将音频分成块处理，可能会占用大量内存
        segment: 设置每个块的分割大小，可以帮助节省显卡内存
        two_stems: 仅分离为两个声部 (例如 "vocals" 将分离为 vocals 和 no_vocals)
        clip_mode: 避免削波的策略，可选 "rescale" 或 "clamp"
        mp3_bitrate: MP3转换的比特率
        mp3_preset: MP3编码器预设，2为最高质量，7为最快速度
        filename: 设置输出文件名，支持变量 {track}、{trackext}、{stem}、{ext}
                 例如: "{track}_{stem}.{ext}" 生成 "歌曲名_vocals.wav"
        jobs: 并行作业数 (0表示自动选择最佳值，通常为CPU核心数)
        verbose: 显示详细输出
    """
    # 构建命令
    cmd = ["demucs"]
    
    # 添加基本参数
    if model_name:
        cmd.extend(["-n", model_name])
    if output_dir:
        cmd.extend(["-o", output_dir])
    if device:
        cmd.extend(["-d", device])
    if shifts:
        cmd.extend(["--shifts", str(shifts)])
    if jobs > 0:  # 只有当jobs大于0时才设置
        cmd.extend(["-j", str(jobs)])
    
    # 添加可选参数
    if two_stems:
        cmd.extend(["--two-stems", two_stems])
    if overlap:
        cmd.extend(["--overlap", str(overlap)])
    if no_split:
        cmd.append("--no-split")
    if segment:
        cmd.extend(["--segment", str(segment)])
    if clip_mode:
        cmd.extend(["--clip-mode", clip_mode])
    if mp3_bitrate:
        cmd.extend(["--mp3-bitrate", str(mp3_bitrate)])
    if mp3_preset:
        cmd.extend(["--mp3-preset", str(mp3_preset)])
    if filename:
        cmd.extend(["--filename", filename])
    if verbose:
        cmd.append("-v")
    
    # 添加音频文件路径
    cmd.append(track_path)
    
    # 执行命令
    print(f"执行命令: {' '.join(cmd)}")
    
    # 当verbose为True时，不捕获输出，让详细信息直接显示在控制台
    if verbose:
        result = subprocess.run(cmd)
    else:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
    # 输出结果
    if result.returncode == 0:
        print("音频分离成功!")
        print(f"分离结果保存在: {os.path.join(output_dir, model_name)}")
        return True
    else:
        print("音频分离失败!")
        if not verbose:  # 只有在非详细模式下才需要打印错误信息
            print(f"错误信息: {result.stderr}")
        return False

def main():
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description="音频分离工具")
    parser.add_argument("track", help="要分离的音频文件路径")
    parser.add_argument("-o", "--output", default="separated", help="输出目录")
    parser.add_argument("-n", "--model", default="htdemucs", help="预训练模型名称")
    parser.add_argument("-d", "--device", default="cuda", choices=["cuda", "cpu"], help="使用的设备")
    parser.add_argument("--shifts", type=int, default=8, help="随机移位数量，增加质量但需要更多时间")
    parser.add_argument("--overlap", type=float, default=0.5, help="分割之间的重叠")
    parser.add_argument("--no-split", action="store_true", help="不将音频分成块处理，可能会占用大量内存")
    parser.add_argument("--segment", type=int, default=7, help="设置每个块的分割大小，可以帮助节省显卡内存")
    parser.add_argument("--two-stems", default="vocals", help="仅分离为两个声部 (例如 vocals)")
    parser.add_argument("--clip-mode", choices=["rescale", "clamp"], default="rescale", help="避免削波的策略: rescale(必要时缩放整个信号)或clamp(硬削波)")
    parser.add_argument("--mp3-bitrate", type=int, default=256, help="MP3转换的比特率")
    parser.add_argument("--mp3-preset", type=int, default=2, choices=[2,3,4,5,6,7], help="MP3编码器预设，2为最高质量，7为最快速度")
    parser.add_argument("--filename", default="{track}_{stem}.{ext}",help="设置输出文件名，支持变量 原文件名{track}、原文件扩展名{trackext}、分离声部{stem}、输出文件扩展名{ext}")
    parser.add_argument("-j", "--jobs", type=int, default=0, help="并行作业数 (0表示自动选择最佳值，通常为CPU核心数)")
    parser.add_argument("-v", "--verbose", default=True, action="store_true", help="显示详细输出")
    
    args = parser.parse_args()
    
    separate_audio(
        args.track,
        output_dir=args.output,
        model_name=args.model,
        device=args.device,
        shifts=args.shifts,
        overlap=args.overlap,
        no_split=args.no_split,
        segment=args.segment,
        two_stems=args.two_stems,
        clip_mode=args.clip_mode,
        mp3_bitrate=args.mp3_bitrate,
        mp3_preset=args.mp3_preset,
        filename=args.filename,
        jobs=args.jobs,
        verbose=args.verbose
    )
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\n执行时间: {execution_time:.2f} 秒")

if __name__ == "__main__":
    main()
