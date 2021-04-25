from PySide2.QtWidgets import QApplication, QFileDialog, QMessageBox
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QPixmap,QImage
from PySide2.QtCore import Signal, QObject,QTimer
import time
import os
import numpy as np
import cv2
import torch
from albumentations import (PadIfNeeded,Normalize,Compose)
import time
import datetime
import math
from Module.parts_seg import CleanU_Net as parts_seg
from albumentations.pytorch.transforms import img_to_tensor
from Module.binary_seg_model import CleanU_Net as binary_seg
from Module.手术器械分割 import file_name, mask_overlay, illum, splitFrames, save_video


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# 让torch判断是否使用GPU

class MySignals(QObject):
    # 定义一种信号，参数是str，即文件的地址
    ms = Signal(str)

model_path_ms = MySignals()
data_path_ms = MySignals()
result_path_ms = MySignals()


class MainWin():  # 主窗口
    def __init__(self):
        super(MainWin, self).__init__()
        # 动态加载主窗口ui界面
        self.ui = QUiLoader().load('ui/main_win.ui')
        self.ui.load_model_btn.clicked.connect(self.load_model)
        self.ui.start_divide_btn.clicked.connect(self.start_divide)
        self.ui.show_result_btn.clicked.connect(self.show_result)
        model_path_ms.ms.connect(self.get_model_path)
        data_path_ms.ms.connect(self.get_data_path)
        result_path_ms.ms.connect(self.get_result_path)

    def load_model(self):
        self.load_model_win = LoadModelWin()
        self.load_model_win.ui.show()

    def get_model_path(self, path):
        text = '成功加载模型！\n模型路径：' + path
        self.ui.load_model_text.setPlainText(text)

    def start_divide(self):
        model_path = self.ui.load_model_text.toPlainText().split('：')[1]
        self.start_divide_win = StartDivideWin(model_path)
        self.start_divide_win.ui.show()

    def get_data_path(self, path):
        text = '成功加载数据！\n数据路径：' + path
        self.ui.data_path_text.setPlainText(text)

    def get_result_path(self, path):
        text = '成功分割数据！\n结果路径：' + path
        self.resultpath= path ##方便传给下面 这里还有点问题 默认是result/test.avi
        self.ui.result_path_text.setPlainText(text)

    def show_result(self):
        self.show_result_win = ShowResultWin(self.resultpath)
        self.show_result_win.ui.show()


class LoadModelWin():  # 加载模型的窗口
    def __init__(self):
        super().__init__()
        self.ui = QUiLoader().load('ui/load_model_win.ui')
        self.ui.browse_btn.clicked.connect(self.get_model_path)
        self.ui.ok_btn.clicked.connect(self.ok)
        self.ui.cancel_btn.clicked.connect(self.cancel)

    def get_model_path(self):
        FileDialog = QFileDialog(self.ui.browse_btn)  # 实例化
        FileDialog.setFileMode(QFileDialog.AnyFile)  # 可以打开任何文件
        model_file, _ = FileDialog.getOpenFileName(self.ui.browse_btn, 'open file', 'model/',
                                                   'model files (*.pth *.pt)')
        # 改变Text里面的文字
        self.ui.model_path_text.setPlainText(model_file)

    def ok(self):
        self.ui.close()
        model_path = self.ui.model_path_text.toPlainText()
        model_path_ms.ms.emit(model_path)

    def cancel(self):
        self.ui.close()
        QMessageBox.warning(self.ui.cancel_btn, "警告", "加载模型失败！", QMessageBox.Yes)


class StartDivideWin():  # 开始分割的窗口
    def __init__(self, path):
        super().__init__()
        self.model_path = path
        self.ui = QUiLoader().load('ui/start_divide_win.ui')
        self.ui.browse_btn_1.clicked.connect(self.get_data_path)
        self.ui.browse_btn_2.clicked.connect(self.get_result_path)
        self.ui.ok_btn.clicked.connect(self.ok)
        self.ui.cancel_btn.clicked.connect(self.cancel)

    def get_data_path(self):
        FileDialog = QFileDialog(self.ui.browse_btn_1)
        FileDialog.setFileMode(QFileDialog.AnyFile)
        type = self.ui.type_combo.currentText()
        data_file = ''
        if type=='image':
            data_file = QFileDialog.getExistingDirectory(self.ui.browse_btn_1, "选择图片文件夹", "./")
        elif type=='video':
            data_file = FileDialog.getOpenFileName(self.ui.browse_btn_1, '打开视频文件', './','video files (*.avi *.mp4)')
        elif type=='nii':
            data_file = FileDialog.getOpenFileName(self.ui.browse_btn_1, 'nii文件', './', 'video files (*.avi *.mp4)')
        self.ui.data_path_text.setPlainText(data_file)

    def get_result_path(self):
        FileDialog = QFileDialog(self.ui.browse_btn_2)
        FileDialog.setFileMode(QFileDialog.AnyFile)
        result_file = QFileDialog.getExistingDirectory(self.ui.browse_btn_2, "选择保存结果文件夹", "./")
        self.ui.result_path_text.setPlainText(result_file)

    def ok(self):
        self.ui.close()
        data_path = self.ui.data_path_text.toPlainText()
        data_path_ms.ms.emit(data_path)
        result_path = self.ui.result_path_text.toPlainText()##保存的路径 不包含文件名
        self.aviname=time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
        result_path_ms.ms.emit(result_path+'/'+self.aviname+'.avi')##传送给主窗口展示的是包含文件名的
        print(result_path+'/'+self.aviname+'.avi')
        # self.ok_win = DividingWin()
        # self.ok_win.ui.show()
        self.ok_win = DividedWin()
        self.ok_win.ui.show()
        self.start_divide()

    def start_divide(self):
        class_color = [[0, 0, 0], [0, 255, 0], [0, 255, 255], [125, 255, 12]]
        t1 = time.time()
        # ********
        seg = "binary"  # 多类别分割，夹子为一类，铰链为一类，柄为一类

        if seg == "parts":
            model = parts_seg(in_channels=3, out_channels=4)
            # model_path = r"model_2_TDSNet.pt"  # 多类别分割模型地址
        elif seg == "binary":  # 二元分割，背景为一类手术器械为一类
            model = binary_seg(in_channels=3, out_channels=2)
            # model_path = r"model_0_TDSNet.pt"  # 二元分割模型地址
        else:
            model = r""
            model_path = r""

        model_path = self.model_path

        state = torch.load(str(model_path), map_location='cpu')
        state = {key.replace('module.', ''): value for key, value in state['model'].items()}
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        # **********
        types = self.ui.type_combo.currentText()
        path = self.ui.data_path_text.toPlainText()  # 要分割的图像或者视频或者nii所在的文件夹地址
        path_save = self.ui.result_path_text.toPlainText()  # 要保存图像的地址
        names = file_name(path)  # path目录下所有文件名字
        factor = 2 ** 5
        # ********

        if types == "image":
            for i, name in enumerate(names):
                t1 = time.time()
                path_single = os.path.join(path, name)
                image = cv2.imdecode(np.fromfile(path_single, dtype=np.uint8), -1)

                h, w, channel = image.shape

                h = math.ceil(h / factor) * factor // 2  # 向上取整，由于模型需要下采样5次图像会变成原来的2的5次方分之一，需要输入图像是2的5次方的倍数
                w = math.ceil(w / factor) * factor // 2
                # print(h, w)
                mask = np.zeros(shape=(h, w))

                image = cv2.resize(image, (w, h))
                image_ori = image
                # image=illum(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                aug = Compose([
                    # PadIfNeeded(min_height=h, min_width=w, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),  # padding到2的5次方的倍数
                    Normalize(p=1)  # 归一化
                ])
                augmented = aug(image=image, mask=mask)
                image = augmented['image']
                image = img_to_tensor(image).unsqueeze(0).to(
                    device)  # torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(DEVICE)  # 图像转为tensor格式
                output = model(image)  # 预测
                seg_mask = (output[0].data.cpu().numpy().argmax(axis=0)).astype(np.uint8)
                t2 = time.time()
                print("time:", (t2 - t1))

                full_mask = np.zeros((h, w, 3))
                for mask_label, sub_color in enumerate(class_color):
                    full_mask[seg_mask == mask_label, 0] = sub_color[2]
                    full_mask[seg_mask == mask_label, 1] = sub_color[1]
                    full_mask[seg_mask == mask_label, 2] = sub_color[0]
                # print(full_mask.max())
                # import matplotlib.pyplot as plt
                # plt.imshow(full_mask)
                # plt.show()

                seg = mask_overlay(image_ori, (full_mask > 0)).astype(np.uint8)
                # seg = mask_overlay((full_mask > 0)).astype(np.uint8)
                # cv2.imshow("seg",seg)
                # cv2.waitKey(0)
                cv2.imwrite(path_save + "/" + str(i).zfill(4) + ".png", seg)

            save_video(path_save, path_save, h, w,self.aviname)  # 可以播放保存的视频展示分割结果

        elif types == "video":
            # 对视频进行分割
            splitFrames(path, path_save)
            #  调用图像分割的方法分割从视频中保存的图像

        elif types == "nii":
            # 对nii3维医学图像进行分割
            import nibabel as nib
            img = np.array(nib.load(path).get_data())
            pass

    def cancel(self):
        self.ui.close()
        QMessageBox.warning(self.ui.cancel_btn, "警告", "未能成功开始分割！", QMessageBox.Yes)


class DividingWin():
    def __init__(self):
        super().__init__()
        self.ui = QUiLoader().load('ui/dividing_win.ui')
        # self.print_run_log  # 不太清楚这么写对不对
        self.ui.mute_run_btn.clicked.connect(self.mute_run)
        self.ui.interrupt_btn.clicked.connect(self.interrupt)

    # def print_run_log(self):
    # 在ui的run_log中打印分割模型运行时候的时间输出或者别的运行输出

    def mute_run(self):  # 后台运行函数
        self.ui.close()
        # 这里需要一段代码，通过判断分割模型运行结束，才打开分割完成的提示窗口
        self.divided_win = DividedWin()
        self.divided_win.ui.show()

    def interrupt(self):
        self.ui.close()
        # 这里需要一段代码来中断分割模型
        QMessageBox.warning(self.ui.interrupt_btn, "警告", "分割被中断！", QMessageBox.Yes)


class DividedWin():  #
    def __init__(self):
        super().__init__()
        self.ui = QUiLoader().load('ui/divided_win.ui')
        self.ui.ok_btn.clicked.connect(self.ok)
        self.ui.cancel_btn.clicked.connect(self.cancel)

    def ok(self):
        # 需要把读取数据的地址和结果地址传回主窗口并在主窗口显示
        self.ui.close()

    def cancel(self):
        self.ui.close()
        QMessageBox.warning(self.ui.cancel_btn, "警告", "分割结果地址未得到！", QMessageBox.Yes)


class ShowResultWin():  # 展示结果窗口
    def __init__(self,resultpath):
        super().__init__()
        self.ui = QUiLoader().load('ui/show_result_win.ui')
        # 这里需要有代码能够去读主窗口中放置结果的路径，并打开存下的视频
        self.file=resultpath
        if not self.file:
            return
        self.ui.LoadingInfo.setText("正在读取请稍后...")
        # 设置时钟
        self.v_timer = QTimer() #self
        # 读取视频
        self.cap = cv2.VideoCapture(self.file)
        if not self.cap:
            print("打开视频失败")
            return
        # 获取视频FPS
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) # 获得码率
        # 获取视频总帧数
        self.total_f = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # 获取视频当前帧所在的帧数
        self.current_f = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        # 设置定时器周期，单位毫秒
        self.v_timer.start(int(1000 / self.fps))
        print("FPS:".format(self.fps))
        # 连接定时器周期溢出的槽函数，用于显示一帧视频
        self.v_timer.timeout.connect(self.show_pic)
        # 连接按钮和对应槽函数，lambda表达式用于传参
        self.ui.play.clicked.connect(self.go_pause)
        self.ui.replay.clicked.connect(self.replay)
        self.ui.back.pressed.connect(lambda: self.last_img(True))
        self.ui.back.clicked.connect(lambda: self.last_img(False))
        self.ui.forward.pressed.connect(lambda: self.next_img(True))
        self.ui.forward.clicked.connect(lambda: self.next_img(False))
        print("init OK")

    # 视频播放
    def show_pic(self):
        # 读取一帧
        success, frame = self.cap.read()
        if success:
            # Mat格式图像转Qt中图像的方法
            show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
            self.ui.Video_Label.setPixmap(QPixmap.fromImage(showImage))
            self.ui.Video_Label.setScaledContents(True)  # 让图片自适应 label 大小

            # 状态栏显示信息
            self.current_f = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            current_t, total_t = self.calculate_time(self.current_f, self.total_f, self.fps)
            self.ui.LoadingInfo.setText("文件名：{}        {}({})".format( self.file, current_t, total_t))

    def calculate_time(self, c_f, t_f, fps):
        total_seconds = int(t_f / fps)
        current_sec = int(c_f / fps)
        c_time = "{}:{}:{}".format(int(current_sec / 3600), int((current_sec % 3600) / 60), int(current_sec % 60))
        t_time = "{}:{}:{}".format(int(total_seconds / 3600), int((total_seconds % 3600) / 60), int(total_seconds % 60))
        return c_time, t_time

    def show_pic_back(self):
        # 获取视频当前帧所在的帧数
        self.current_f = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        # 设置下一次帧为当前帧-2
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_f-2)
        # 读取一帧
        success, frame = self.cap.read()
        if success:
            show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
            self.ui.Video_Label.setPixmap(QPixmap.fromImage(showImage))

            # 状态栏显示信息
            current_t, total_t = self.calculate_time(self.current_f-1, self.total_f, self.fps)
            self.ui.LoadingInfo.setText("文件名：{}        {}({})".format( self.file, current_t, total_t))
        # 快退

    def replay(self):
        self.v_timer = QTimer()  # self
        self.ui.play.setText("暂停")
        # 读取视频
        self.cap = cv2.VideoCapture(self.file)
        if not self.cap:
            print("打开视频失败")
            return
        # 获取视频FPS
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)  # 获得码率
        # 获取视频总帧数
        self.total_f = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # 获取视频当前帧所在的帧数
        self.current_f = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        # 设置定时器周期，单位毫秒
        self.v_timer.start(int(1000 / self.fps))
        print("FPS:".format(self.fps))
        # 连接定时器周期溢出的槽函数，用于显示一帧视频
        self.v_timer.timeout.connect(self.show_pic)
        # 连接按钮和对应槽函数，lambda表达式用于传参
        print("play OK")

    def last_img(self, t):
        self.ui.play.setText("暂停")
        if t:
            # 断开槽连接
            self.v_timer.timeout.disconnect(self.show_pic)
            # 连接槽连接
            self.v_timer.timeout.connect(self.show_pic_back)
            self.v_timer.start(int(1000 / self.fps) / 2)
        else:
            self.v_timer.timeout.disconnect(self.show_pic_back)
            self.v_timer.timeout.connect(self.show_pic)
            self.v_timer.start(int(1000 / self.fps))
        # 快进

    def next_img(self, t):
        self.ui.play.setText("暂停")
        if t:
            self.v_timer.start(int(1000 / self.fps) / 2)  # 快进
        else:
            self.v_timer.start(int(1000 / self.fps))
# 暂停播放

    def go_pause(self):
        if  self.ui.play.text() == "暂停":
            self.v_timer.stop()
            self.ui.play.setText("播放")
        elif self.ui.play.text() == "播放":
            self.v_timer.start(int(1000/self.fps))
            self.ui.play.setText("暂停")


def main():
    app = QApplication([])
    start = MainWin()
    start.ui.show()
    app.exec_()


if __name__ == '__main__':
    main()
