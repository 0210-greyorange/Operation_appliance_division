from PySide2.QtWidgets import QApplication, QFileDialog, QMessageBox
from PySide2.QtUiTools import QUiLoader


class MainWin():  # 主窗口
    def __init__(self):
        super(MainWin, self).__init__()
        # 动态加载主窗口ui界面
        self.ui = QUiLoader().load('ui/main_win.ui')
        self.ui.load_model_btn.clicked.connect(self.load_model)
        self.ui.start_divide_btn.clicked.connect(self.start_divide)
        self.ui.show_result_btn.clicked.connect(self.show_result)

    def load_model(self):
        self.load_model_win = LoadModelWin()
        self.load_model_win.ui.show()

    def start_divide(self):
        self.start_divide_win = StartDivideWin()
        self.start_divide_win.ui.show()

    def show_result(self):
        self.show_result_win = ShowResultWin()
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
        model_file, _ = FileDialog.getOpenFileName(self.ui.browse_btn, 'open file', './',
                                                   'model files (*.pth *.pt)')
        # 改变Text里面的文字
        self.ui.model_path_text.setPlainText(model_file)

    def ok(self):
        self.ui.close()
        # 这里需要一段代码把模型地址传回主窗口，并在主窗口显示

    def cancel(self):
        self.ui.close()
        QMessageBox.warning(self.ui.cancel_btn, "警告", "加载模型失败！", QMessageBox.Yes)


class StartDivideWin():  # 开始分割的窗口
    def __init__(self):
        super().__init__()
        self.ui = QUiLoader().load('ui/start_divide_win.ui')
        self.ui.browse_btn_1.clicked.connect(self.get_data_path)
        self.ui.browse_btn_2.clicked.connect(self.get_result_path)
        self.ui.ok_btn.clicked.connect(self.ok)
        self.ui.cancel_btn.clicked.connect(self.cancel)

    def get_data_path(self):
        FileDialog = QFileDialog(self.ui.browse_btn_1)
        FileDialog.setFileMode(QFileDialog.AnyFile)
        # 分情况：如果self.ui.type_combo是image，则data_file是一个文件夹的路径，如果是video或者nii则是一个文件的路径
        data_file, _ = FileDialog.getOpenFileName(self.ui.browse_btn_1, 'open file', './',
                                                  'model files (*.pth *.pt)')
        self.ui.data_path_text.setPlainText(data_file)

    def get_result_path(self):
        FileDialog = QFileDialog(self.ui.browse_btn_2)
        FileDialog.setFileMode(QFileDialog.AnyFile)
        # 下面这个result_file需要是一个文件夹的地址，最终分割的结果就存在这个文件夹里
        result_file, _ = FileDialog.getOpenFileName(self.ui.browse_btn_1, 'open file', './',
                                                    'model files (*.pth *.pt)')
        self.ui.result_path_text.setPlainText(result_file)

    def ok(self):
        self.ui.close()
        # 这里开始调用手术器械分割模型的测试3（或者也可以在DividingWin窗口里直接调用）
        self.ok_win = DividingWin()
        self.ok_win.ui.show()

    def cancel(self):
        self.ui.close()
        QMessageBox.warning(self.ui.cancel_btn, "警告", "未能成功开始分割！", QMessageBox.Yes)


class DividingWin():
    def __init__(self):
        super().__init__()
        self.ui = QUiLoader().load('ui/dividing.ui')
        self.print_run_log  # 不太清楚这么写对不对
        self.ui.mute_run_btn.clicked.connect(self.mute_run)
        self.ui.interrupt_btn.clicked.connect(self.interrupt)

    def print_run_log(self):
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
        self.ui = QUiLoader().load('ui/divided.ui')
        self.ui.ok_btn.clicked.connect(self.ok)
        self.ui.cancel_btn.clicked.connect(self.cancel)

    def ok(self):
        # 需要把读取数据的地址和结果地址传回主窗口并在主窗口显示
        self.ui.close()

    def cancel(self):
        self.ui.close()
        QMessageBox.warning(self.ui.cancel_btn, "警告", "分割结果地址未得到！", QMessageBox.Yes)


class ShowResultWin():  # 展示结果窗口
    def __init__(self):
        super().__init__()
        self.ui = QUiLoader().load('ui/show_result_win.ui')
        # 这里需要有代码能够去读主窗口中放置结果的路径，并打开存下的视频


def main():
    app = QApplication([])
    start = MainWin()
    start.ui.show()
    app.exec_()


if __name__ == '__main__':
    main()
