import os
import subprocess
import argparse
import time
import glob
import concurrent.futures
import threading
import multiprocessing
import queue
import signal
import sys

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

# 用于多线程处理时的打印锁，防止输出混乱
print_lock = threading.Lock()

# 全局变量，用于优雅退出
shutdown_event = threading.Event()

# 信号处理函数
def signal_handler(sig, frame):
    print("\n收到中断信号，正在优雅退出...")
    shutdown_event.set()
    
# 只在主线程中注册信号处理函数
# 这样可以避免在Streamlit等非主线程环境中出错
try:
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
except ValueError:
    # 如果不在主线程中，忽略错误
    pass

# 检测可用的GPU数量
def get_available_gpus():
    try:
        # 尝试使用CUDA检测GPU
        result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
        if result.returncode == 0:
            # 计算GPU数量
            gpu_count = len([line for line in result.stdout.split('\n') if 'GPU ' in line])
            return max(1, gpu_count)  # 至少返回1
        return 1  # 默认返回1
    except Exception:
        return 1  # 如果出错，默认返回1

# 获取推荐的线程数
def get_recommended_threads():
    # 获取CPU核心数
    cpu_count = multiprocessing.cpu_count()
    # 推荐线程数 = CPU核心数 - 2（至少为1）
    return max(1, cpu_count - 2)

def process_single_file(file_info, output_dir, model_name, device, shifts, overlap, no_split, segment, 
                       two_stems, clip_mode, mp3_bitrate, mp3_preset, filename, jobs, verbose, gpu_id=None):
    """
    处理单个音频文件的工作函数，用于多线程处理
    
    参数:
        file_info: 包含文件路径和索引的元组 (index, file_path, total_files)
        其他参数与separate_audio函数相同
    
    返回:
        成功处理返回True，否则返回False
    """
    index, file_path, total_files = file_info
    
    with print_lock:
        print(f"\n[{index}/{total_files}] 处理文件: {file_path}")
    
    # 检查是否收到退出信号
    if shutdown_event.is_set():
        with print_lock:
            print(f"跳过处理文件: {file_path} (收到退出信号)")
        return False
        
    try:
        # 如果指定了GPU ID，则修改设备参数
        actual_device = device
        if gpu_id is not None and device == "cuda":
            actual_device = f"cuda:{gpu_id}"
            
        success = separate_audio(
            file_path,
            output_dir=output_dir,
            model_name=model_name,
            device=actual_device,
            shifts=shifts,
            overlap=overlap,
            no_split=no_split,
            segment=segment,
            two_stems=two_stems,
            clip_mode=clip_mode,
            mp3_bitrate=mp3_bitrate,
            mp3_preset=mp3_preset,
            filename=filename,
            jobs=jobs,
            verbose=verbose
        )
        
        with print_lock:
            if success:
                print(f"✓ 成功处理: {file_path}")
            else:
                print(f"✗ 处理失败: {file_path}")
        
        return success
    except Exception as e:
        with print_lock:
            print(f"✗ 处理文件 '{file_path}' 时出错: {str(e)}")
        return False

def batch_process_directory(directory_path, output_dir="separated", model_name="htdemucs", device="cuda", 
                        shifts=8, overlap=0.5, no_split=False, segment=7, two_stems="vocals", 
                        clip_mode="rescale", mp3_bitrate=256, mp3_preset=2, filename=None, 
                        jobs=0, verbose=False, extensions=None, max_workers=None, use_gpu_allocation=True):
    """
    递归处理指定目录下的所有音频文件
    
    参数:
        directory_path: 要处理的目录路径
        extensions: 要处理的音频文件扩展名列表，默认为['.mp3', '.wav', '.flac', '.ogg', '.m4a']
        max_workers: 最大并行工作线程数，默认为None（根据系统自动选择）
        其他参数与separate_audio函数相同
    
    返回:
        成功处理的文件数量
    """
    if extensions is None:
        extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']
    
    # 确保扩展名都是小写且带点
    extensions = [ext.lower() if ext.startswith('.') else '.' + ext.lower() for ext in extensions]
    
    # 获取所有音频文件
    audio_files = []
    for ext in extensions:
        # 使用glob递归查找所有匹配的文件
        pattern = os.path.join(directory_path, f'**/*{ext}')
        audio_files.extend(glob.glob(pattern, recursive=True))
    
    # 排序以确保处理顺序一致
    audio_files.sort()
    
    if not audio_files:
        print(f"在目录 '{directory_path}' 中没有找到支持的音频文件")
        return 0
    
    total_files = len(audio_files)
    print(f"找到 {total_files} 个音频文件需要处理")
    
    # 如果max_workers为1，则使用单线程处理
    if max_workers == 1:
        print("使用单线程顺序处理模式")
        success_count = 0
        for i, file_path in enumerate(audio_files, 1):
            print(f"\n[{i}/{total_files}] 处理文件: {file_path}")
            
            try:
                success = separate_audio(
                    file_path,
                    output_dir=output_dir,
                    model_name=model_name,
                    device=device,
                    shifts=shifts,
                    overlap=overlap,
                    no_split=no_split,
                    segment=segment,
                    two_stems=two_stems,
                    clip_mode=clip_mode,
                    mp3_bitrate=mp3_bitrate,
                    mp3_preset=mp3_preset,
                    filename=filename,
                    jobs=jobs,
                    verbose=verbose
                )
                
                if success:
                    success_count += 1
                    print(f"✓ 成功处理: {file_path}")
                else:
                    print(f"✗ 处理失败: {file_path}")
            except Exception as e:
                print(f"✗ 处理文件 '{file_path}' 时出错: {str(e)}")
        
        print(f"\n批量处理完成: 成功 {success_count}/{total_files}")
        return success_count
    
    # 重置退出事件
    shutdown_event.clear()
    
    # 如果未指定max_workers，使用推荐线程数
    if max_workers is None:
        max_workers = get_recommended_threads()
        print(f"自动设置线程数为: {max_workers} (CPU核心数-2)")
    
    # 检测GPU数量并决定是否分配GPU
    gpu_count = 1
    gpu_allocation = False
    if device == "cuda" and use_gpu_allocation:
        gpu_count = get_available_gpus()
        gpu_allocation = gpu_count > 1
        if gpu_allocation:
            print(f"检测到 {gpu_count} 个GPU，将自动分配任务到不同GPU")
    
    # 使用多线程处理
    print(f"使用多线程并行处理模式 (线程数: {max_workers})")
    
    # 准备文件信息列表，包含索引和总数
    file_infos = [(i+1, file_path, total_files) for i, file_path in enumerate(audio_files)]
    
    # 使用线程池进行并行处理
    success_count = 0
    task_queue = queue.Queue()
    
    # 将所有任务放入队列
    for file_info in file_infos:
        task_queue.put(file_info)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = []
        
        # 创建工作线程
        for i in range(max_workers):
            # 为每个线程分配GPU（如果有多个GPU）
            gpu_id = i % gpu_count if gpu_allocation else None
            
            # 提交任务
            futures.append(executor.submit(
                worker_thread, 
                task_queue, 
                output_dir, 
                model_name, 
                device, 
                shifts, 
                overlap, 
                no_split, 
                segment, 
                two_stems, 
                clip_mode, 
                mp3_bitrate, 
                mp3_preset, 
                filename, 
                jobs, 
                verbose,
                gpu_id
            ))
        
        # 收集结果
        for future in concurrent.futures.as_completed(futures):
            try:
                # 每个工作线程返回它成功处理的文件数
                success_count += future.result()
            except Exception as e:
                with print_lock:
                    print(f"工作线程发生异常: {str(e)}")
    
    print(f"\n批量处理完成: 成功 {success_count}/{total_files}")
    return success_count

# 工作线程函数
def worker_thread(task_queue, output_dir, model_name, device, shifts, overlap, no_split, segment, 
               two_stems, clip_mode, mp3_bitrate, mp3_preset, filename, jobs, verbose, gpu_id=None):
    """
    工作线程函数，从队列中获取任务并处理
    
    返回:
        成功处理的文件数量
    """
    success_count = 0
    
    while not shutdown_event.is_set():
        try:
            # 非阻塞方式获取任务，允许检查退出信号
            try:
                file_info = task_queue.get(block=True, timeout=0.5)
            except queue.Empty:
                # 队列为空，退出循环
                break
                
            # 处理文件
            if process_single_file(
                file_info, 
                output_dir, 
                model_name, 
                device, 
                shifts, 
                overlap, 
                no_split, 
                segment, 
                two_stems, 
                clip_mode, 
                mp3_bitrate, 
                mp3_preset, 
                filename, 
                jobs, 
                verbose,
                gpu_id
            ):
                success_count += 1
                
            # 标记任务完成
            task_queue.task_done()
            
        except Exception as e:
            with print_lock:
                print(f"工作线程处理任务时发生异常: {str(e)}")
    
    return success_count

def main():
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description="音频分离工具")
    parser.add_argument("input", help="要分离的音频文件路径或包含音频文件的目录路径")
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
    parser.add_argument("--batch", action="store_true", help="批量处理模式，将input视为目录并递归处理其中的所有音频文件")
    parser.add_argument("--extensions", nargs="+", default=['.mp3', '.wav', '.flac', '.ogg', '.m4a'], 
                      help="在批量处理模式下要处理的文件扩展名，例如: --extensions mp3 wav flac")
    parser.add_argument("--threads", type=int, default=None, 
                      help="批量处理时的并行线程数，默认为CPU核心数-2。设置为1禁用多线程处理")
    parser.add_argument("--no-gpu-allocation", action="store_true",
                      help="禁用GPU自动分配，所有线程使用相同的GPU设备")
    
    args = parser.parse_args()
    
    # 确保扩展名格式正确
    extensions = [ext if ext.startswith('.') else '.' + ext for ext in args.extensions]
    
    # 判断是单文件处理还是批量处理
    if args.batch or os.path.isdir(args.input):
        thread_info = f"{args.threads} 个线程" if args.threads else "自动选择线程数"
        print(f"批量处理模式: 处理目录 '{args.input}' 中的所有音频文件 (使用{thread_info})")
        batch_process_directory(
            args.input,
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
            verbose=args.verbose,
            extensions=extensions,
            max_workers=args.threads,
            use_gpu_allocation=not args.no_gpu_allocation
        )
    else:
        print(f"单文件处理模式: 处理文件 '{args.input}'")
        separate_audio(
            args.input,
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
    print(f"\n总执行时间: {execution_time:.2f} 秒")

if __name__ == "__main__":
    main()
