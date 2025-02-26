#!/usr/bin/env python3
import os
import json
import shutil
import subprocess
from pathlib import Path

class OfflinePackageBuilder:
    def __init__(self, config_file='tools/package_config.json'):
        self.config_file = config_file
        self.base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.offline_dir = self.base_dir / 'offline_packages'
        self.python_dir = self.offline_dir / 'python_packages'
        self.ubuntu_dir = self.offline_dir / 'ubuntu_packages'
        self.load_config()

    def load_config(self):
        """加载配置文件"""
        with open(self.config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

    def setup_directories(self):
        """创建必要的目录"""
        print("创建目录结构...")
        for dir_path in [self.offline_dir, self.python_dir, self.ubuntu_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def download_python_packages(self):
        """下载Python包"""
        print("下载Python包...")
        requirements_file = self.offline_dir / 'requirements.txt'
        
        # 写入requirements.txt
        with open(requirements_file, 'w', encoding='utf-8') as f:
            for package in self.config['python_packages']:
                f.write(f"{package}\n")
        
        # 下载包
        cmd = f"pip download -r {requirements_file} -d {self.python_dir}"
        subprocess.run(cmd, shell=True, check=True)

    def download_ubuntu_packages(self):
        """下载Ubuntu包"""
        print("下载Ubuntu包...")
        packages = ' '.join(self.config['ubuntu_packages'])
        cmd = f"cd {self.ubuntu_dir} && apt-get download {packages}"
        subprocess.run(cmd, shell=True, check=True)

    def generate_readme(self):
        """生成README文件"""
        print("生成README文件...")
        readme_content = f"""# {self.config['readme_template']['title']}

{self.config['readme_template']['description']}

## 1. 安装Ubuntu系统依赖

所有必需的Ubuntu包都已下载到 `ubuntu_packages` 目录中。按以下步骤安装：

```bash
# 进入ubuntu_packages目录
cd ubuntu_packages

# 安装所有.deb包
sudo dpkg -i *.deb

# 如果出现依赖问题，运行
sudo apt-get install -f
```

## 2. 安装Python包

所有Python包都已下载到 `python_packages` 目录中。

1. 确保目标机器上已安装Python {self.config['readme_template']['python_version']}
2. 在目标机器上执行以下命令：

```bash
# 安装所有依赖包
pip install --no-index --find-links=./python_packages/ -r requirements.txt
```

## 注意事项
- 确保Python版本与包版本兼容（推荐Python {self.config['readme_template']['python_version']}）
- 如果遇到权限问题，可能需要使用sudo或在虚拟环境中安装
- 如果使用虚拟环境，请确保先激活环境再安装包

## 包含的依赖

### Ubuntu系统包：
{chr(10).join('- ' + pkg for pkg in self.config['ubuntu_packages'])}

### Python包：
{chr(10).join('- ' + pkg for pkg in self.config['python_packages'])}
"""
        
        with open(self.offline_dir / 'README.md', 'w', encoding='utf-8') as f:
            f.write(readme_content)

    def create_archive(self):
        """创建最终的压缩包"""
        print("创建压缩包...")
        archive_name = 'AEA_offline_packages.tar.gz'
        cmd = f"cd {self.base_dir} && tar -czf {archive_name} offline_packages/"
        subprocess.run(cmd, shell=True, check=True)

    def clean_old_files(self):
        """清理旧的打包文件"""
        print("清理旧文件...")
        if self.offline_dir.exists():
            shutil.rmtree(self.offline_dir)
        
        archive = self.base_dir / 'AEA_offline_packages.tar.gz'
        if archive.exists():
            archive.unlink()

    def build(self):
        """执行完整的打包流程"""
        try:
            print("开始打包流程...")
            self.clean_old_files()
            self.setup_directories()
            self.download_python_packages()
            self.download_ubuntu_packages()
            self.generate_readme()
            self.create_archive()
            print("打包完成！生成文件：AEA_offline_packages.tar.gz")
        except Exception as e:
            print(f"打包过程中出错：{str(e)}")
            raise

if __name__ == '__main__':
    builder = OfflinePackageBuilder()
    builder.build()
