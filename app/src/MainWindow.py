from PyQt6.QtCore import QFile, QIODevice
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import QFileDialog, QMainWindow
from os import getcwd


class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        self.setMinimumSize(500, 500)
        self.setWindowTitle("Tweet Sentiment Analysis")
        self.createMenuBar([
            ["Upload", "Ctrl+O", "Upload a file", self.openFile],
            ["Clear", "Ctrl+N", "Clear the output", print],
            ["Exit", "", "", print]
        ])

    def createMenuBar(self, actions: list):
        action_bar = self.menuBar()

        for i in range(len(actions)):
            action = QAction(actions[i][0], self)

            action.setShortcut(QKeySequence(actions[i][1]))
            action.setStatusTip(actions[i][2])
            action.triggered.connect(actions[i][3])
            action_bar.addAction(action)
  
    def openFile(self):
        file_name = QFileDialog.getOpenFileName(self, "Open File", getcwd(), "*csv")[0]
        file = QFile(file_name)

        if file.open(QIODevice.OpenModeFlag.ReadOnly):
            print(file.readAll())
            file.close()
