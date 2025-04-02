import sys
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
# 修复导入错误 - 使用兼容的matplotlib后端
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QFileDialog, QLabel, 
                           QComboBox, QGroupBox, QTextEdit, 
                           QScrollArea, QDoubleSpinBox, QSizePolicy,
                           QGridLayout, QFrame)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from events_guess import predict_audio_events

# 设置matplotlib使用Qt后端
import matplotlib
matplotlib.use('QtAgg')  # 使用通用的QtAgg后端，它会自动选择适合的Qt版本
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class AudioEventAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('音频事件分析系统')
        
        # 设置窗口大小为屏幕尺寸的80%
        screen = QApplication.primaryScreen().geometry()
        width = int(screen.width() * 0.8)
        height = int(screen.height() * 0.8)
        self.setGeometry(0, 0, width, height)
        
        # 将窗口移动到屏幕中央
        self.move(int((screen.width() - width) / 2),
                 int((screen.height() - height) / 2))
        
        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 创建主布局
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        main_widget.setLayout(layout)
        
        # 创建左侧控制面板
        control_panel = self.create_control_panel()
        control_panel.setMaximumWidth(300)
        layout.addWidget(control_panel, stretch=2)
        
        # 创建中间图表区域
        charts_panel = self.create_charts_panel()
        layout.addWidget(charts_panel, stretch=7)
        
        # 创建右侧结果面板
        results_panel = self.create_results_panel()
        results_panel.setMaximumWidth(350)
        layout.addWidget(results_panel, stretch=3)
        
        # 初始化变量
        self.audio_path = None
        self.y = None
        self.sr = None
        
    def create_control_panel(self):
        panel = QFrame()
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        panel.setFrameShadow(QFrame.Shadow.Raised)
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)
        panel.setLayout(layout)
        
        # 标题
        title_label = QLabel("控制面板")
        title_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # 文件选择
        file_group = QGroupBox("文件选择")
        file_layout = QVBoxLayout()
        file_layout.setSpacing(8)
        self.file_label = QLabel("未选择文件")
        self.file_label.setWordWrap(True)
        self.file_button = QPushButton("选择音频文件")
        self.file_button.setMinimumHeight(30)
        self.file_button.clicked.connect(self.load_audio_file)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.file_button)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # 模型选择
        model_group = QGroupBox("模型设置")
        model_layout = QVBoxLayout()
        model_layout.setSpacing(8)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["随机森林 (RF)", "XGBoost (XGB)"])
        self.model_combo.setMinimumHeight(30)
        # 添加一个信号连接，当模型选择改变时调整置信度
        self.model_combo.currentIndexChanged.connect(self.adjust_confidence_threshold)
        model_layout.addWidget(QLabel("选择模型:"))
        model_layout.addWidget(self.model_combo)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # 参数设置
        params_group = QGroupBox("参数设置")
        params_layout = QGridLayout()
        params_layout.setVerticalSpacing(10)
        params_layout.setHorizontalSpacing(5)
        
        # 窗口大小
        params_layout.addWidget(QLabel("窗口大小 (秒):"), 0, 0)
        self.window_size = QDoubleSpinBox()
        self.window_size.setRange(0.5, 5.0)
        self.window_size.setValue(2.0)
        self.window_size.setSingleStep(0.1)
        self.window_size.setMinimumHeight(30)
        params_layout.addWidget(self.window_size, 0, 1)
        
        # 滑动步长
        params_layout.addWidget(QLabel("滑动步长 (秒):"), 1, 0)
        self.hop_length = QDoubleSpinBox()
        self.hop_length.setRange(0.1, 2.0)
        self.hop_length.setValue(1.0)
        self.hop_length.setSingleStep(0.1)
        self.hop_length.setMinimumHeight(30)
        params_layout.addWidget(self.hop_length, 1, 1)
        
        # 置信度阈值
        params_layout.addWidget(QLabel("置信度阈值:"), 2, 0)
        self.confidence = QDoubleSpinBox()
        self.confidence.setRange(0.0, 1.0)
        self.confidence.setValue(0.6)
        self.confidence.setSingleStep(0.05)
        self.confidence.setMinimumHeight(30)
        params_layout.addWidget(self.confidence, 2, 1)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # 检测按钮
        self.detect_button = QPushButton("开始检测")
        self.detect_button.setMinimumHeight(40)
        self.detect_button.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.detect_button.clicked.connect(self.run_detection)
        self.detect_button.setEnabled(False)
        layout.addWidget(self.detect_button)
        
        layout.addStretch()
        return panel
        
    def create_charts_panel(self):
        """创建中间的图表显示区域"""
        panel = QScrollArea()
        content = QWidget()
        layout = QGridLayout()
        layout.setSpacing(10)
        content.setLayout(layout)
        
        # 波形图 (左上)
        self.waveform_group = QGroupBox("波形图")
        waveform_layout = QVBoxLayout()
        self.waveform_canvas = self.create_plot_canvas()
        waveform_layout.addWidget(self.waveform_canvas)
        self.waveform_group.setLayout(waveform_layout)
        layout.addWidget(self.waveform_group, 0, 0)
        
        # 梅尔频谱图 (右上)
        self.mel_group = QGroupBox("梅尔频谱图")
        mel_layout = QVBoxLayout()
        self.mel_canvas = self.create_plot_canvas()
        mel_layout.addWidget(self.mel_canvas)
        self.mel_group.setLayout(mel_layout)
        layout.addWidget(self.mel_group, 0, 1)
        
        # MFCC特征图 (左下)
        self.mfcc_group = QGroupBox("MFCC特征图")
        mfcc_layout = QVBoxLayout()
        self.mfcc_canvas = self.create_plot_canvas()
        mfcc_layout.addWidget(self.mfcc_canvas)
        self.mfcc_group.setLayout(mfcc_layout)
        layout.addWidget(self.mfcc_group, 1, 0)
        
        # RMS能量图 (右下)
        self.rms_group = QGroupBox("RMS能量图")
        rms_layout = QVBoxLayout()
        self.rms_canvas = self.create_plot_canvas()
        rms_layout.addWidget(self.rms_canvas)
        self.rms_group.setLayout(rms_layout)
        layout.addWidget(self.rms_group, 1, 1)
        
        panel.setWidget(content)
        panel.setWidgetResizable(True)
        return panel

    def create_results_panel(self):
        """创建右侧的检测结果显示区域"""
        panel = QFrame()
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        panel.setFrameShadow(QFrame.Shadow.Raised)
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        panel.setLayout(layout)
        
        # 结果标题
        title_label = QLabel("检测结果")
        title_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # 结果文本框
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFont(QFont("Consolas", 10))
        layout.addWidget(self.result_text, stretch=1)
        
        return panel

    def create_plot_canvas(self):
        figure = plt.figure(figsize=(5, 3.5))
        canvas = FigureCanvas(figure)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        return canvas
        
    def adjust_confidence_threshold(self):
        """根据选择的模型自动调整置信度阈值"""
        # 获取当前选中的模型
        current_model = self.model_combo.currentText()
        
        # 根据模型类型设置不同的默认置信度阈值
        if "XGB" in current_model:
            # XGBoost模型使用较高的置信度阈值
            self.confidence.setValue(0.8)
        else:
            # 随机森林模型使用较低的置信度阈值
            self.confidence.setValue(0.6)
            
        # 可以在这里显示一个简短的提示信息告知用户
        self.result_text.append(f"已根据{current_model}模型特性，将置信度阈值调整为{self.confidence.value()}")
        
    def load_audio_file(self):
        dialog = QFileDialog()
        dialog.setWindowTitle("选择音频文件")
        dialog.setNameFilter("音频文件 (*.wav *.mp3)")
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        
        if dialog.exec() == QFileDialog.DialogCode.Accepted:
            file_path = dialog.selectedFiles()[0]
            self.audio_path = file_path
            self.file_label.setText(os.path.basename(file_path))
            self.detect_button.setEnabled(True)
            try:
                self.y, self.sr = librosa.load(file_path, sr=None)
                self.plot_audio_features()
                self.result_text.setText(f"已加载音频文件：{os.path.basename(file_path)}\n"
                                       f"采样率：{self.sr} Hz\n"
                                       f"持续时间：{len(self.y)/self.sr:.2f} 秒")
            except Exception as e:
                self.result_text.setText(f"加载音频文件失败：{str(e)}")
                
    def plot_audio_features(self):
        if self.y is None or self.sr is None:
            return
            
        # 波形图
        self.waveform_canvas.figure.clear()
        ax = self.waveform_canvas.figure.add_subplot(111)
        ax.plot(np.linspace(0, len(self.y)/self.sr, len(self.y)), self.y)
        ax.set_title('波形图')
        ax.set_xlabel('时间 (秒)')
        ax.set_ylabel('振幅')
        ax.grid(True, linestyle='--', alpha=0.7)
        self.waveform_canvas.figure.tight_layout()
        self.waveform_canvas.draw()
        
        # 梅尔频谱图
        self.mel_canvas.figure.clear()
        ax = self.mel_canvas.figure.add_subplot(111)
        mel_spect = librosa.feature.melspectrogram(y=self.y, sr=self.sr)
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
        img = librosa.display.specshow(mel_spect_db, y_axis='mel', 
                                     x_axis='time', ax=ax)
        self.mel_canvas.figure.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title('梅尔频谱图')
        self.mel_canvas.figure.tight_layout()
        self.mel_canvas.draw()
        
        # MFCC特征图
        self.mfcc_canvas.figure.clear()
        ax = self.mfcc_canvas.figure.add_subplot(111)
        mfccs = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=13)
        img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
        self.mfcc_canvas.figure.colorbar(img, ax=ax)
        ax.set_title('MFCC特征图')
        self.mfcc_canvas.figure.tight_layout()
        self.mfcc_canvas.draw()

        # RMS能量图
        self.rms_canvas.figure.clear()
        ax = self.rms_canvas.figure.add_subplot(111)
        rms = librosa.feature.rms(y=self.y)
        ax.plot(np.linspace(0, len(self.y)/self.sr, len(rms.T)), rms.T)
        ax.set_title('RMS能量图')
        ax.set_xlabel('时间 (秒)')
        ax.set_ylabel('RMS能量')
        ax.grid(True, linestyle='--', alpha=0.7)
        self.rms_canvas.figure.tight_layout()
        self.rms_canvas.draw()
        
    def run_detection(self):
        if not self.audio_path:
            return
            
        try:
            self.result_text.setText("正在进行检测，请稍候...")
            QApplication.processEvents()  # 强制更新界面
            
            model_type = "xgb" if "XGB" in self.model_combo.currentText() else "rf"
            model_path = "models/audio_event_model_xgboost.pkl" if model_type == "xgb" else "models/audio_event_model_segments.pkl"
            
            events = predict_audio_events(
                self.audio_path,
                window_size=self.window_size.value(),
                hop_length=self.hop_length.value(),
                confidence_threshold=self.confidence.value(),
                model_type=model_type,
                model_path=model_path
            )
            
            # 显示检测结果
            result_text = "检测结果：\n\n"
            
            if events:
                for i, (event, start, end, confidence) in enumerate(events, 1):
                    result_text += f"事件 {i}:\n"
                    result_text += f"类型: {event}\n"
                    result_text += f"置信度: {confidence:.2%}\n"
                    result_text += f"时间段: {start:.1f}s - {end:.1f}s\n"
                    result_text += f"持续时间: {end-start:.1f}s\n"
                    result_text += "-" * 30 + "\n"
                
                # 添加摘要信息
                result_text += f"\n统计摘要:\n"
                result_text += f"检测到的事件总数: {len(events)}\n"
                total_duration = sum(end-start for _, start, end, _ in events)
                result_text += f"事件总持续时间: {total_duration:.1f}s\n"
                audio_duration = len(self.y)/self.sr if self.y is not None and self.sr is not None else 0
                result_text += f"音频总时长: {audio_duration:.1f}s\n"
            else:
                result_text = "未检测到显著事件"
                
            self.result_text.setText(result_text)
            
        except Exception as e:
            self.result_text.setText(f"检测失败：{str(e)}")

def main():
    app = QApplication(sys.argv)
    window = AudioEventAnalyzer()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
