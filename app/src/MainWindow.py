from PyQt6.QtWidgets import (
    QFileDialog, QMainWindow, QVBoxLayout,
    QWidget, QPushButton, QLabel, QTableWidget, QTableWidgetItem,
    QComboBox, QHBoxLayout
)
from PyQt6.QtGui import QBrush, QColor, QFont
from os import getcwd
import pandas as pd  

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setMinimumSize(900, 700)
        self.setWindowTitle("X Sentiment Analysis")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # background and text colors
        self.setStyleSheet("""
            QMainWindow { background-color: #F4F6F9; color: #333333; font-family: 'Helvetica Neue', Arial, sans-serif; }
            QLabel { font-size: 14px; font-weight: bold; color: #333333; }
            QComboBox, QTableWidget { font-size: 14px; }
        """)

        # Button Config
        self.upload_train_button = QPushButton("Upload Training Data")
        self.upload_test_button = QPushButton("Upload Test Data")
        self.clean_data_button = QPushButton("Clean Data")
        self.classify_button = QPushButton("Classify Test Tweets")
        self.save_button = QPushButton("Save Annotations")

        self.algorithm_selector = QComboBox()
        self.algorithm_selector.addItems(["KNN-Classifier", "Naive Bayes"])
        self.sentiments = {0: "Negative", 2: "Neutral", 4: "Positive"}

        # Drop Box
        self.algorithm_selector.setStyleSheet("""
        QComboBox {
            background-color: #000000;  # Black background for the combo box
            border: 1px solid #B0B0B0;
            border-radius: 8px;
            padding: 12px;
            font-size: 14px;
            color: white;  # White text for the combo box itself
        }
        QComboBox:hover {
            border-color: #4CAF50;
        }
        QComboBox::drop-down {
            background-color: #000000;  # Black background for the dropdown
            border: none;
            padding: 5px;
            border-radius: 0;
        }
        QComboBox QAbstractItemView {
            background-color: #000000;  # Black background for dropdown items
            color: white;  # White text for dropdown items
            border: 1px solid #B0B0B0;
            selection-background-color: #D1E7DD;
            font-size: 14px;
        }
        QComboBox::item:selected {
            background-color: #D1E7DD;
        }
        QComboBox::item {
            padding: 10px;
        }
        QComboBox::down-arrow {
            width: 12px;
            height: 12px;
        }
        """)


        # Table Config
        self.tweet_table = QTableWidget()
        self.tweet_table.setColumnCount(2)
        self.tweet_table.setHorizontalHeaderLabels(["Tweet", "Sentiment"])
        self.tweet_table.setStyleSheet("""
            QTableWidget {
                background-color: #FFFFFF;
                color: #333333;
                border: 1px solid #E0E0E0;
                gridline-color: #E0E0E0;
                font-size: 14px;
                border-radius: 8px;
                padding: 10px;
                selection-background-color: #D1E7DD;  # Highlight selected row
            }

            QHeaderView::section {
                background-color: #4CAF50;
                color: white;
                padding: 12px;
                font-size: 16px;
                font-weight: bold;
                border: none;
                border-radius: 8px 8px 0 0;
            }

            QTableWidget::item {
                padding: 12px;
                border: 1px solid #E0E0E0;
                border-radius: 5px;
                text-align: left;  # Align text to the left for better readability
            }

            QTableWidget::item:selected {
                background-color: #D1E7DD;
                color: #333333;
            }

            QTableWidget::item:!selected:hover {
                background-color: #F1F1F1;  # Hover effect for non-selected items
            }

            QTableWidget::corner {
                background-color: #F1F1F1;
                border: none;
            }

            QTableWidget::item:focus {
                border: 1px solid #4CAF50;  # Highlight item when focused
            }
        """)

        self.performance_label = QLabel("Algorithm Performance: Not evaluated")

        # Layout
        self.layout.addWidget(self.upload_train_button)
        self.layout.addWidget(self.upload_test_button)
        self.layout.addWidget(self.clean_data_button)
        self.layout.addWidget(self.tweet_table)
        self.layout.addWidget(self.performance_label)
        primary_button_style = """
            QPushButton { 
                background-color: #4CAF50; 
                color: white; 
                font-weight: bold; 
                padding: 12px; 
                border-radius: 8px; 
                font-size: 16px;
            }
            QPushButton:hover { background-color: #45a049; }
        """
        self.upload_train_button.setStyleSheet(primary_button_style)
        self.upload_test_button.setStyleSheet(primary_button_style)
        self.clean_data_button.setStyleSheet(primary_button_style)
        self.classify_button.setStyleSheet(primary_button_style)
        self.save_button.setStyleSheet(primary_button_style)

        # Bottom Config
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(20)
        bottom_layout.addWidget(QLabel("Choose Algorithm:"))
        bottom_layout.addWidget(self.algorithm_selector)
        bottom_layout.addWidget(self.classify_button)
        bottom_layout.addWidget(self.save_button)
        bottom_layout_widget = QWidget()
        bottom_layout_widget.setLayout(bottom_layout)
        bottom_layout_widget.setStyleSheet("""
        QLabel {
            color: white;  # White text for the "Choose Algorithm" label
        }
        background-color: #F1F1F1; 
        padding: 20px; 
        border-radius: 15px;
        margin-top: 20px;
        """)
        self.layout.addWidget(bottom_layout_widget)

        self.upload_train_button.clicked.connect(self.upload_training_file)
        self.upload_test_button.clicked.connect(self.upload_test_file)
        self.clean_data_button.clicked.connect(self.clean_data)
        self.classify_button.clicked.connect(self.classify_tweets)
        self.save_button.clicked.connect(self.save_annotations)

    def upload_training_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Training Data", getcwd(), "CSV Files (*.csv)")
        if file_name:
            self.train_data = pd.read_csv(file_name)
            self.populate_table(self.train_data)

    def upload_test_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Test Data", getcwd(), "CSV Files (*.csv)")
        if file_name:
            self.test_data = pd.read_csv(file_name)
            self.populate_table(self.test_data)

    def populate_table(self, data):
        self.tweet_table.setRowCount(len(data))
        self.tweet_table.setColumnWidth(0, 600)
        self.tweet_table.setColumnWidth(1, 120)

        for row in range(len(data)):
            tweet = data.iloc[row, 5]  
            target = data.iloc[row, 0]  

            color = QColor("grey") 
            if target == 0:
                color = QColor("#520413")  
            elif target == 4:
                color = QColor("#4CAF50") 
        
            tweet_item = QTableWidgetItem(tweet)
            sentiment_item = QTableWidgetItem(self.sentiments.get(target, "Unknown"))

            tweet_item.setBackground(QBrush(color))
            sentiment_item.setBackground(QBrush(color))

            self.tweet_table.setItem(row, 0, tweet_item)
            self.tweet_table.setItem(row, 1, sentiment_item)

    def clean_data(self):
        print("Data cleaned.")

    def classify_tweets(self):
        selected_algorithm = self.algorithm_selector.currentText()
        print(f"Classifying using {selected_algorithm}")

    def save_annotations(self):
        print("Annotations saved.")
