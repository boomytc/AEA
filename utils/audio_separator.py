import os
import subprocess
import argparse

def separate_audio(
    track_path,
    output_dir="separated",
    model_name="htdemucs",
    signature=None,
    repo=None,
    device="cuda",
    shifts=5,
    overlap=0.25,
    no_split=False,
    segment=None,
    two_stems="vocals",
    int24=False,
    float32=False,
    clip_mode="rescale",
    flac=False,
    mp3=False,
    mp3_bitrate=256,
    mp3_preset=2,
    filename=None,
    jobs=6,
    verbose=False
):
    """
    使用 Demucs 进行音频分离
    
    参数:
        track_path: 音频文件路径
        output_dir: 输出目录
        model_name: 预训练模型名称
        signature: 本地训练的XP签名
        repo: 包含所有预训练模型的文件夹
        device: 使用的设备 (cuda 或 cpu)
        shifts: 随机移位数量 (增加质量但需要更多时间)
        overlap: 分割之间的重叠
        no_split: 不将音频分成块处理，可能会占用大量内存
        segment: 设置每个块的分割大小，可以帮助节省显卡内存
        two_stems: 仅分离为两个声部 (例如 "vocals" 将分离为 vocals 和 no_vocals)
        int24: 将WAV输出保存为24位
        float32: 将WAV输出保存为float32格式 (文件大小增加2倍)
        clip_mode: 避免削波的策略，可选 "rescale" 或 "clamp"
        flac: 将输出WAV转换为FLAC格式
        mp3: 将输出转换为MP3格式
        mp3_bitrate: MP3转换的比特率
        mp3_preset: MP3编码器预设，2为最高质量，7为最快速度
        filename: 设置输出文件名，支持变量 {track}、{trackext}、{stem}、{ext}
        jobs: 并行作业数
        verbose: 显示详细输出
    """
    # 构建命令
    cmd = ["demucs"]
    
    # 添加基本参数
    if model_name:
        cmd.extend(["-n", model_name])
    if signature:
        cmd.extend(["-s", signature])
    if repo:
        cmd.extend(["--repo", repo])
    if output_dir:
        cmd.extend(["-o", output_dir])
    if device:
        cmd.extend(["-d", device])
    if shifts:
        cmd.extend(["--shifts", str(shifts)])
    if jobs:
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
    if int24:
        cmd.append("--int24")
    if float32:
        cmd.append("--float32")
    if clip_mode:
        cmd.extend(["--clip-mode", clip_mode])
    if flac:
        cmd.append("--flac")
    if mp3:
        cmd.append("--mp3")
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
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # 输出结果
    if result.returncode == 0:
        print("音频分离成功!")
        print(f"分离结果保存在: {os.path.join(output_dir, model_name)}")
        return True
    else:
        print("音频分离失败!")
        print(f"错误信息: {result.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="音频分离工具")
    parser.add_argument("track", help="要分离的音频文件路径")
    parser.add_argument("-o", "--output", default="separated", help="输出目录")
    parser.add_argument("-n", "--model", default="htdemucs", help="预训练模型名称")
    parser.add_argument("-s", "--signature", help="本地训练的XP签名")
    parser.add_argument("--repo", help="包含所有预训练模型的文件夹")
    parser.add_argument("-d", "--device", default="cuda", choices=["cuda", "cpu"], help="使用的设备")
    parser.add_argument("--shifts", type=int, default=10, help="随机移位数量，增加质量但需要更多时间")
    parser.add_argument("--overlap", type=float, default=0.25, help="分割之间的重叠")
    parser.add_argument("--no-split", action="store_true", help="不将音频分成块处理，可能会占用大量内存")
    parser.add_argument("--segment", type=float, help="设置每个块的分割大小，可以帮助节省显卡内存")
    parser.add_argument("--two-stems", default="vocals", help="仅分离为两个声部 (例如 vocals)")
    parser.add_argument("--int24", action="store_true", help="将WAV输出保存为24位")
    parser.add_argument("--float32", action="store_true", help="将WAV输出保存为float32格式 (文件大小增加2倍)")
    parser.add_argument("--clip-mode", choices=["rescale", "clamp"], default="rescale", 
                        help="避免削波的策略: rescale(必要时缩放整个信号)或clamp(硬削波)")
    parser.add_argument("--flac", action="store_true", help="将输出WAV转换为FLAC格式")
    parser.add_argument("--mp3", action="store_true", help="将输出转换为MP3格式")
    parser.add_argument("--mp3-bitrate", type=int, default=256, help="MP3转换的比特率")
    parser.add_argument("--mp3-preset", type=int, default=2, choices=[2,3,4,5,6,7], 
                        help="MP3编码器预设，2为最高质量，7为最快速度")
    parser.add_argument("--filename", help="设置输出文件名，支持变量 {track}、{trackext}、{stem}、{ext}")
    parser.add_argument("-j", "--jobs", type=int, default=8, help="并行作业数，增加内存使用但多核处理更快")
    parser.add_argument("-v", "--verbose", action="store_true", help="显示详细输出")
    
    args = parser.parse_args()
    
    separate_audio(
        args.track,
        output_dir=args.output,
        model_name=args.model,
        signature=args.signature,
        repo=args.repo,
        device=args.device,
        shifts=args.shifts,
        overlap=args.overlap,
        no_split=args.no_split,
        segment=args.segment,
        two_stems=args.two_stems,
        int24=args.int24,
        float32=args.float32,
        clip_mode=args.clip_mode,
        flac=args.flac,
        mp3=args.mp3,
        mp3_bitrate=args.mp3_bitrate,
        mp3_preset=args.mp3_preset,
        filename=args.filename,
        jobs=args.jobs,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()
