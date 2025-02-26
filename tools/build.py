import os
import shutil
import subprocess
import platform

def build_project():
    """构建项目"""
    print("开始构建项目...")
    
    # 清理之前的构建文件
    if os.path.exists('build'):
        shutil.rmtree('build')
    if os.path.exists('dist'):
        shutil.rmtree('dist')
    
    # 使用PyInstaller构建
    subprocess.run(['pyinstaller', 'tools/audio_analyzer.spec'])
    
    # 创建发布目录
    release_dir = 'release'
    if os.path.exists(release_dir):
        shutil.rmtree(release_dir)
    os.makedirs(release_dir)
    
    # 打印dist目录内容，用于调试
    print("\ndist 目录内容:")
    for item in os.listdir('dist'):
        print(f"- {item}")
    
    # 复制可执行文件到发布目录
    executables = ['特征提取工具', '事件检测工具', '模型训练工具']
    for exe in executables:
        # 在dist目录中查找匹配的文件
        found = False
        for item in os.listdir('dist'):
            if item.startswith(exe):
                src = os.path.join('dist', item)
                dst = os.path.join(release_dir, item)
                print(f"复制: {src} -> {dst}")
                shutil.copy(src, dst)
                # 在Linux下设置可执行权限
                if platform.system() != 'Windows':
                    os.chmod(dst, 0o755)
                found = True
                break
        
        if not found:
            print(f"警告：找不到可执行文件 {exe}")
    
    # 创建必要的目录
    os.makedirs(f'{release_dir}/models', exist_ok=True)
    os.makedirs(f'{release_dir}/temp', exist_ok=True)
    
    # 复制说明文档
    with open(f'{release_dir}/使用说明.txt', 'w', encoding='utf-8') as f:
        f.write("""音频事件检测系统使用说明：

1. 特征提取工具：
   - 用于单独提取音频特征
   - 使用方法：将音频文件拖放到程序窗口或作为参数运行

2. 事件检测工具：
   - 用于检测音频中的事件
   - 使用方法：将音频文件拖放到程序窗口或作为参数运行
   - 需要 models 目录中存在训练好的模型文件

3. 模型训练工具：
   - 用于训练新的模型
   - 需要准备好标注数据文件 data_list.txt
   - 训练好的模型将保存在 models 目录下

注意事项：
- 请确保 models 目录中存在训练好的模型文件
- temp 目录用于临时文件存储，请勿删除
""")
    
    print(f"\n构建完成！发布文件在 {release_dir} 目录中。")

if __name__ == '__main__':
    build_project() 