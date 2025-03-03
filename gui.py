import sys
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QFileDialog, QLabel, 
                           QComboBox, QSlider, QGroupBox, QTextEdit, 
                           QScrollArea, QDoubleSpinBox)
from PyQt5.QtCore import Qt
from events_guess import predict_audio_events
import matplotlib
matplotlib.use('Qt5Agg')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class AudioEventAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('音频事件分析系统')
        
        # 设置窗口大小为屏幕尺寸的80%
        screen = QApplication.primaryScreen().geometry()
        width = int(screen.width() * 0.8)  # 增加到80%
        height = int(screen.height() * 0.6)
        self.setGeometry(0, 0, width, height)
        
        # 将窗口移动到屏幕中央
        self.move(int((screen.width() - width) / 2),
                 int((screen.height() - height) / 2))
        
        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 创建主布局
        layout = QHBoxLayout()
        main_widget.setLayout(layout)
        
        # 创建左侧控制面板
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel, stretch=1)
        
        # 创建中间图表区域
        charts_panel = self.create_charts_panel()
        layout.addWidget(charts_panel, stretch=7)
        
        # 创建右侧结果面板
        results_panel = self.create_results_panel()
        layout.addWidget(results_panel, stretch=1)
        
        # 初始化变量
        self.audio_path = None
        self.y = None
        self.sr = None
        
    def create_control_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # 文件选择
        file_group = QGroupBox("文件选择")
        file_layout = QVBoxLayout()
        self.file_label = QLabel("未选择文件")
        self.file_button = QPushButton("选择音频文件")
        self.file_button.clicked.connect(self.load_audio_file)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.file_button)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # 模型选择
        model_group = QGroupBox("模型设置")
        model_layout = QVBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(["随机森林 (RF)", "XGBoost (XGB)"])
        model_layout.addWidget(QLabel("选择模型:"))
        model_layout.addWidget(self.model_combo)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # 参数设置
        params_group = QGroupBox("参数设置")
        params_layout = QVBoxLayout()
        
        # 窗口大小
        window_layout = QVBoxLayout()
        window_layout.addWidget(QLabel("窗口大小 (秒):"))
        self.window_size = QDoubleSpinBox()
        self.window_size.setRange(0.5, 5.0)
        self.window_size.setValue(2.0)
        self.window_size.setSingleStep(0.1)
        window_layout.addWidget(self.window_size)
        params_layout.addLayout(window_layout)
        
        # 滑动步长
        hop_layout = QVBoxLayout()
        hop_layout.addWidget(QLabel("滑动步长 (秒):"))
        self.hop_length = QDoubleSpinBox()
        self.hop_length.setRange(0.1, 2.0)
        self.hop_length.setValue(1.5)
        self.hop_length.setSingleStep(0.1)
        hop_layout.addWidget(self.hop_length)
        params_layout.addLayout(hop_layout)
        
        # 置信度阈值
        conf_layout = QVBoxLayout()
        conf_layout.addWidget(QLabel("置信度阈值:"))
        self.confidence = QDoubleSpinBox()
        self.confidence.setRange(0.0, 1.0)
        self.confidence.setValue(0.6)
        self.confidence.setSingleStep(0.05)
        conf_layout.addWidget(self.confidence)
        params_layout.addLayout(conf_layout)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # 检测按钮
        self.detect_button = QPushButton("开始检测")
        self.detect_button.clicked.connect(self.run_detection)
        self.detect_button.setEnabled(False)
        layout.addWidget(self.detect_button)
        
        layout.addStretch()
        return panel
        
    def create_charts_panel(self):
        """创建中间的图表显示区域"""
        panel = QScrollArea()
        content = QWidget()
        layout = QVBoxLayout()
        content.setLayout(layout)
        
        # 创建上下两个行布局
        top_row = QHBoxLayout()
        bottom_row = QHBoxLayout()
        
        # 波形图 (左上)
        self.waveform_group = QGroupBox("波形图")
        waveform_layout = QVBoxLayout()
        self.waveform_canvas = self.create_plot_canvas()
        waveform_layout.addWidget(self.waveform_canvas)
        self.waveform_group.setLayout(waveform_layout)
        top_row.addWidget(self.waveform_group)
        
        # 梅尔频谱图 (右上)
        self.mel_group = QGroupBox("梅尔频谱图")
        mel_layout = QVBoxLayout()
        self.mel_canvas = self.create_plot_canvas()
        mel_layout.addWidget(self.mel_canvas)
        self.mel_group.setLayout(mel_layout)
        top_row.addWidget(self.mel_group)
        
        # MFCC特征图 (左下)
        self.mfcc_group = QGroupBox("MFCC特征图")
        mfcc_layout = QVBoxLayout()
        self.mfcc_canvas = self.create_plot_canvas()
        mfcc_layout.addWidget(self.mfcc_canvas)
        self.mfcc_group.setLayout(mfcc_layout)
        bottom_row.addWidget(self.mfcc_group)
        
        # RMS能量图 (右下)
        self.rms_group = QGroupBox("RMS能量图")
        rms_layout = QVBoxLayout()
        self.rms_canvas = self.create_plot_canvas()
        rms_layout.addWidget(self.rms_canvas)
        self.rms_group.setLayout(rms_layout)
        bottom_row.addWidget(self.rms_group)
        
        layout.addLayout(top_row)
        layout.addLayout(bottom_row)
        
        panel.setWidget(content)
        panel.setWidgetResizable(True)
        return panel

    def create_results_panel(self):
        """创建右侧的检测结果显示区域"""
        panel = QScrollArea()
        content = QWidget()
        layout = QVBoxLayout()
        content.setLayout(layout)
        
        # 检测结果显示
        self.result_group = QGroupBox("检测结果")
        result_layout = QVBoxLayout()
        
        # 结果文本框
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        result_layout.addWidget(self.result_text)
        
        self.result_group.setLayout(result_layout)
        layout.addWidget(self.result_group)
        
        panel.setWidget(content)
        panel.setWidgetResizable(True)
        return panel

    def create_plot_canvas(self):
        # 增加图表尺寸
        figure = plt.figure(figsize=(8, 4))  # 调整为更大的尺寸
        canvas = FigureCanvas(figure)
        return canvas
        
    def load_audio_file(self):
        dialog = QFileDialog(self, "选择音频文件")
        dialog.setNameFilter("音频文件 (*.wav *.mp3)")
        dialog.setViewMode(QFileDialog.Detail)
        
        # 设置对话框大小为屏幕尺寸的60%
        screen = QApplication.primaryScreen().geometry()
        dialog.resize(int(screen.width() * 0.5), int(screen.height() * 0.4))
        
        # 将对话框移动到屏幕中央
        dialog.move(
            int((screen.width() - dialog.width()) / 2),
            int((screen.height() - dialog.height()) / 2)
        )
        
        if dialog.exec_() == QFileDialog.Accepted:
            file_path = dialog.selectedFiles()[0]
            self.audio_path = file_path
            self.file_label.setText(os.path.basename(file_path))
            self.detect_button.setEnabled(True)
            try:
                self.y, self.sr = librosa.load(file_path, sr=None)
                self.plot_audio_features()
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
        self.mel_canvas.draw()
        
        # MFCC特征图
        self.mfcc_canvas.figure.clear()
        ax = self.mfcc_canvas.figure.add_subplot(111)
        mfccs = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=13)
        img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
        self.mfcc_canvas.figure.colorbar(img, ax=ax)
        ax.set_title('MFCC特征图')
        self.mfcc_canvas.draw()

        # RMS能量图
        self.rms_canvas.figure.clear()
        ax = self.rms_canvas.figure.add_subplot(111)
        rms = librosa.feature.rms(y=self.y)
        ax.plot(np.linspace(0, len(self.y)/self.sr, len(rms.T)), rms.T)
        ax.set_title('RMS能量图')
        ax.set_xlabel('时间 (秒)')
        ax.set_ylabel('RMS能量')
        self.rms_canvas.draw()
        
    def run_detection(self):
        if not self.audio_path:
            return
            
        try:
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
            
            # 合并显示检测结果
            result_text = "检测结果：\n\n"
            
            if events:
                for i, (event, start, end, confidence) in enumerate(events, 1):
                    result_text += f"事件 {i}:\n"
                    result_text += f"类型: {event}\n"
                    result_text += f"置信度: {confidence:.2%}\n"
                    result_text += f"时间段: {start:.1f}s - {end:.1f}s\n"
                    result_text += f"持续时间: {end-start:.1f}s\n"
                    result_text += "-" * 30 + "\n"
            else:
                result_text = "未检测到显著事件"
                
            self.result_text.setText(result_text)
            
        except Exception as e:
            self.result_text.setText(f"检测失败：{str(e)}")

def main():
    app = QApplication(sys.argv)
    window = AudioEventAnalyzer()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
