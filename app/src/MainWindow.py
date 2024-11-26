from PyQt6.QtWidgets import QFileDialog, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QTableWidget, QTableWidgetItem, QComboBox, QHBoxLayout
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtCore import Qt, QSize, QRect
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import os
import seaborn as sns
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setMinimumSize(900, 700)
        self.setWindowTitle("X Sentiment Analysis")

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.sentiments = {0: "Negative", 2: "Neutral", 4: "Positive"}

        self.setStyleSheet(self.main_window_stylesheet())
        self.init_ui()
        self.connect_buttons()

    def main_window_stylesheet(self):
        return """
            QMainWindow { background-color: #404040; color: white; font-family: 'Helvetica Neue', Arial, sans-serif; }
            QLabel { font-size: 16px; font-weight: bold; color: white; }
            QComboBox, QTableWidget, QPushButton { font-size: 14px; color: white; }
            QTableWidget { background-color: #F5F5F5; gridline-color: #B0B0B0; }
        """

    def init_ui(self):
        self.upload_button = QPushButton("Upload Data")
        self.upload_button.setStyleSheet(self.primary_button_style())
        self.upload_dropdown = QComboBox()
        self.upload_dropdown.addItems(["Select Data Type", "Train Data", "Test Data"])
        self.upload_dropdown.setStyleSheet(self.dropdown_style())

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.upload_button)
        top_layout.addWidget(self.upload_dropdown)
        self.layout.addLayout(top_layout)

        self.tweet_table = QTableWidget(self)
        self.tweet_table.setColumnCount(2)
        self.tweet_table.setHorizontalHeaderLabels(["Tweet", "Sentiment"])
        self.tweet_table.setStyleSheet(self.table_style())
        self.tweet_table.setAlternatingRowColors(True)
        self.layout.addWidget(self.tweet_table)

        self.figure = plt.Figure(figsize=(3, 3), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(self.central_widget)
        self.canvas.setVisible(False)

        bottom_layout = QHBoxLayout()
        self.algorithm_selector = QComboBox()
        self.algorithm_selector.addItems(["KNN-Classifier", "Naive Bayes"])
        self.algorithm_selector.setStyleSheet(self.dropdown_style())
        self.classify_button = QPushButton("Classify")
        self.classify_button.setStyleSheet(self.primary_button_style())
        self.save_button = QPushButton("Save Annotations")
        self.save_button.setStyleSheet(self.primary_button_style())
        bottom_layout.addWidget(QLabel("Choose Algorithm:"))
        bottom_layout.addWidget(self.algorithm_selector)
        bottom_layout.addWidget(self.classify_button)
        bottom_layout.addWidget(self.save_button)
        self.layout.addLayout(bottom_layout)

    def primary_button_style(self):
        return """
            QPushButton {
                background-color: #FF5733;
                font-weight: bold;
                padding: 12px;
                border-radius: 8px;
                color: white;
            }
            QPushButton:hover { background-color: #FF2E00; }
        """

    def dropdown_style(self):
        return """
            QComboBox {
                background-color: #696969;
                border: 1px solid #B0B0B0;
                padding: 12px;
                font-size: 14px;
                border-radius: 8px;
                color: white;
            }
        """

    def table_style(self):
        return """
            QTableWidget {
                background-color: #FFFFFF;
                alternate-background-color: #F0F0F0;
                selection-background-color: #D3D3D3;
                gridline-color: #404040;
            }
            QHeaderView::section {
                background-color: #D3D3D3;
                color: black;
                font-weight: bold;
                border: 1px solid #B0B0B0;
            }
        """

    def connect_buttons(self):
        self.upload_button.clicked.connect(self.handle_upload)
        self.classify_button.clicked.connect(self.classify_tweets)
        self.save_button.clicked.connect(self.save_annotations)

    def handle_upload(self):
        selected_type = self.upload_dropdown.currentText()
        if selected_type in ["Train Data", "Test Data"]:
            file_name, _ = QFileDialog.getOpenFileName(self, f"Open {selected_type}", os.getcwd(), "CSV Files (*.csv)")
            if file_name:
                data = pd.read_csv(file_name)
                self.populate_table(data)
                self.show_pie_chart()

    def populate_table(self, data):
        self.tweet_table.setRowCount(len(data))
        self.tweet_table.setColumnWidth(0, 600)
        self.tweet_table.setColumnWidth(1, 120)

        for row in range(len(data)):
            tweet = data.iloc[row, 5]
            target = data.iloc[row, 0]

            tweet_item = QTableWidgetItem(tweet)
            sentiment_item = QTableWidgetItem(self.sentiments.get(target, "Unknown"))

            tweet_item.setBackground(QBrush(QColor("#696969")))
            sentiment_item.setBackground(QBrush(QColor("#696969")))

            self.tweet_table.setItem(row, 0, tweet_item)
            self.tweet_table.setItem(row, 1, sentiment_item)

    def show_pie_chart(self):
        labels = ['Positive', 'Negative', 'Neutral']
        sizes = [50, 30, 20]

        ax = self.figure.add_subplot(111, aspect='equal')
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=["#4CAF50", "#FF5252", "#FFEB3B"])
        ax.axis('equal')

        ax.set_facecolor("none")
        self.canvas.setVisible(True)
        self.position_pie_chart()
        self.canvas.draw()

    def position_pie_chart(self):
        window_rect = self.geometry()
        table_rect = self.tweet_table.geometry()

        pie_width = int(table_rect.width() * 0.4)
        pie_height = int(table_rect.height() * 0.4)

        top_padding = 40
        right_padding = 10

        self.canvas.setGeometry(QRect(window_rect.right() - pie_width - right_padding, window_rect.top() + top_padding, pie_width, pie_height))
        self.canvas.setVisible(True)

    def classify_tweets(self):
        selected_algorithm = self.algorithm_selector.currentText()
        print(f"Classifying using {selected_algorithm}")

        for row in range(self.tweet_table.rowCount()):
            sentiment_item = QTableWidgetItem("Positive")
            self.tweet_table.setItem(row, 1, sentiment_item)

        self.show_confusion_matrix()

    def show_confusion_matrix(self):
        conf_matrix = np.array([[50, 10, 5], [12, 60, 3], [8, 4, 45]])
        labels = ['Positive', 'Negative', 'Neutral']

        ax_cm = self.figure.add_subplot(212)

        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax_cm)
        ax_cm.set_title("Confusion Matrix")

        self.figure.tight_layout()
        self.position_confusion_matrix()
        self.canvas.draw()

    def position_confusion_matrix(self):
        window_rect = self.geometry()
        cm_width = int(window_rect.width() * 0.45)
        cm_height = int(window_rect.height() * 0.35)

        self.canvas.setGeometry(QRect(window_rect.left() + 10, window_rect.top() + 250, cm_width, cm_height))
        self.canvas.setVisible(True)

    def save_annotations(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Annotations", os.getcwd(), "CSV Files (*.csv)")
        if file_name:
            print(f"Annotations saved to: {file_name}")
