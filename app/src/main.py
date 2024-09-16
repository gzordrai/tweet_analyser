from PyQt6.QtWidgets import QApplication, QWidget
from sys import argv

if __name__ == "__main__":
    app = QApplication(argv)
    window = QWidget()

    window.show()
    app.exec()
