from PyQt6.QtWidgets import QFileDialog,QProgressBar, QDialog, QMessageBox, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QTableWidget, QTableWidgetItem, QComboBox, QHBoxLayout
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QSize, QRect
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import os
from datetime import datetime
import json
import seaborn as sns
import numpy as np
import requests
from PyQt6.QtWidgets import QErrorMessage

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setMinimumSize(900, 700)
        self.setWindowTitle("Twitter Sentiment Analysis")

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.right_layout = QVBoxLayout()  
        self.main_layout = QHBoxLayout(self.central_widget)  
        self.main_layout.addLayout(self.layout) 
        self.main_layout.addLayout(self.right_layout) 


        self.sentiments = {'0': "Negative", '2': "Neutral", '4': "Positive", '-1' : "UnAnnotated"}
        self.write_sentiments = {"Negative" : '0', "Neutral" : '2', "Positive" : '4'}
        self.api_classifer = {"KNN-Classifier" : "knn", "Naive Bayes" : "bayes"}

        self.setStyleSheet(self.main_window_stylesheet())
        self.init_ui()
        self.connect_buttons()
        self.confusion_window = ConfusionMatrixWindow(self)

    def main_window_stylesheet(self):
        return """
            QMainWindow { background-color: #404040; color: white; font-family: 'Helvetica Neue', Arial, sans-serif; }
            QLabel { font-size: 16px; font-weight: bold; color: white; }
            QComboBox, QTableWidget, QPushButton { font-size: 14px; color: white; }
            QTableWidget { background-color: #F5F5F5; gridline-color: #B0B0B0; }
        """

    

    def init_ui(self):
        self.top_layout = QHBoxLayout()  
        self.upload_button = QPushButton("Upload Data")
        self.upload_button.setStyleSheet(self.primary_button_style())
        self.upload_dropdown = QComboBox()
        self.upload_dropdown.clear()
        self.tweet_table = QTableWidget(self)
        self.upload_dropdown.addItems(["Annotated Dataset", "UnAnnotated Dataset"])
        self.upload_dropdown.setStyleSheet(self.dropdown_style())

        self.tweet_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        self.top_layout.addWidget(self.upload_button)
        self.top_layout.addWidget(self.upload_dropdown)

        action_layout = QHBoxLayout() 
        self.model_parameters_label = QLabel("KNN Value:", self)
        self.algorithm_selector = QComboBox()
        self.algorithm_selector.clear()
        self.algorithm_selector.addItems(["KNN-Classifier", "Naive Bayes"])
        self.algorithm_selector.setStyleSheet(self.dropdown_style())

        action_layout.addWidget(QLabel("Choose Algorithm:"))
        action_layout.addWidget(self.algorithm_selector)
        self.model_parameters_selector = QComboBox()
        self.model_parameters_selector.setStyleSheet(self.dropdown_style())
        self.model_parameters_selector.setFixedWidth(70)  

        action_layout.addWidget(self.model_parameters_label)
        action_layout.addWidget(self.model_parameters_selector)
        

        self.classify_button = QPushButton("Classify")
        self.classify_button.setStyleSheet(self.primary_button_style())
        self.save_button = QPushButton("Save Annotations")
        self.save_button.setStyleSheet(self.primary_button_style())
        
        action_layout.addWidget(self.classify_button)
        action_layout.addWidget(self.save_button)
        # action_layout.addLayout(self.radio_layout)

        self.layout.addLayout(self.top_layout)  
        self.layout.addLayout(action_layout) 

        self.tweet_table.setColumnCount(2)
        self.tweet_table.setHorizontalHeaderLabels(["Tweet", "Sentiment"])
        self.tweet_table.setStyleSheet(self.table_style())
        self.tweet_table.setAlternatingRowColors(True)
        self.layout.addWidget(self.tweet_table)


        self.figure = plt.Figure(figsize=(3, 3), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(self.central_widget)
        self.canvas.setVisible(False)

        self.update_model_parameters_dropdown(self.algorithm_selector.currentText())

        self.algorithm_selector.currentTextChanged.connect(
            lambda: self.update_model_parameters_dropdown(self.algorithm_selector.currentText())
        )


    def update_model_parameters_dropdown(self, selected_algorithm):
        self.model_parameters_selector.clear()
        if selected_algorithm == "KNN-Classifier":
            self.model_parameters_selector.addItems([str(i) for i in range(1, 11, 2)])
            self.model_parameters_label.setText("knn value:")
        elif selected_algorithm == "Naive Bayes":
            self.model_parameters_selector.addItems([str(i) for i in range(1, 6)])
            self.model_parameters_label.setText("ngram value:")
        self.model_parameters_selector.setCurrentIndex(0)


    def connect_buttons(self):
        self.upload_button.clicked.connect(self.handle_upload)
        self.classify_button.clicked.connect(self.classify_tweets)
        self.save_button.clicked.connect(self.save_annotations)

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





    def populate_table(self, data, selected_type, sentiment_index, tweet_index):
        self.tweet_table.setRowCount(len(data) - 300)
        self.tweet_table.setColumnCount(2)
        self.tweet_table.setColumnWidth(0, 600)
        self.tweet_table.setColumnWidth(1, 120)

        for row in range(len(data)):
            tweet = data.iloc[row, tweet_index]  
            sentiment = data.iloc[row, sentiment_index] 
            tweet_item = QTableWidgetItem(tweet)
            tweet_item.setBackground(QBrush(QColor("#696969")))
            tweet_item.setFlags(tweet_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.tweet_table.setItem(row, 0, tweet_item)

          
            combo = QComboBox()
            combo.addItems(["Negative", "Neutral", "Positive", "UnAnnotated"])

    
            sentiment_text = self.sentiments.get(str(sentiment), "UnAnnotated")
            combo.setCurrentText(sentiment_text)
            combo.currentTextChanged.connect(lambda text, row=row: self.update_sentiment_in_dataframe(row, text))

            self.tweet_table.setCellWidget(row, 1, combo)
        
        
        self.show_pie_chart(data)

    def update_sentiment_in_dataframe(self, row, selected_text):

        sentiment_value = self.write_sentiments.get(selected_text, selected_text)
        self.modified_data.iloc[row, 0] = sentiment_value  
        
        sentiment_item = QTableWidgetItem(selected_text)
        sentiment_item.setBackground(QBrush(QColor("#696969")))
        sentiment_item.setFlags(sentiment_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.tweet_table.setItem(row, 1, sentiment_item)


    def set_combobox_for_sentiment(self):
        for row in range(self.tweet_table.rowCount()):
            sentiment_item = self.tweet_table.item(row, 1)
            if sentiment_item:
                combo = QComboBox()
                combo.addItems(["Negative", "Neutral", "Positive", "UnAnnotated"])
                combo.setCurrentText(sentiment_item.text())
                self.tweet_table.setCellWidget(row, 1, combo)


    def handle_upload(self):
        selected_type = self.upload_dropdown.currentText()
        if selected_type in ["Annotated Dataset", "UnAnnotated Dataset"]:

            self.tweet_table.setRowCount(0) 
            self.tweet_table.setColumnCount(2)  
            self.right_layout.removeWidget(self.canvas)  
            self.canvas.setVisible(False)
            file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)")

            if file_path:
                try:
                    self.show_progress_dialog("Loading file, please wait...")
                    self.uploaded_file = file_path
                    data = pd.read_csv(file_path, header=0)
                    data.iloc[:, 0] = data.iloc[:, 0].fillna("-1").astype(str)
                    if data.shape[1] > 5:
                        self.modified_data = data.copy()
                        self.populate_table(data, selected_type, 0, 5)
                    else:
                        QMessageBox.warning(self, "Invalid File", "CSV must have at least 6 columns to extract 'Tweet' and 'Sentiment'.")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Could not read file: {e}")
                finally:
                    self.hide_progress_dialog()
                


    def on_sentiment_change(self, selected_text, row):
        sentiment_value = self.write_sentiments.get(selected_text, 0)
        self.modified_data.iloc[row, 0] = sentiment_value  

    def save_annotations(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Annotations", os.getcwd(), "CSV Files (*.csv)")
        if file_name:
            try:
                self.modified_data.to_csv(file_name, index=False)
                QMessageBox.information(self, "Success", f"Annotations saved to {file_name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save annotations: {e}")


    def classify_tweets(self):
        if not self.upload_dropdown.currentText() in ["Annotated Dataset", "UnAnnotated Dataset"]:
            self.show_error("Please select a dataset type (Train/Test).")
            return
        
        if not self.algorithm_selector.currentText():
            self.show_error("Please select a classifier algorithm.")
            return

        if not hasattr(self, 'uploaded_file'):
            self.show_error("Please upload a file before classifying.")
            return

        dataset_type = self.upload_dropdown.currentText().split()[0].lower()
        classifier = self.algorithm_selector.currentText()
        model_param = self.model_parameters_selector.currentText()
        param_label = self.model_parameters_label.text()

        api_url = f"http://localhost:8000/dataset/{dataset_type}"

        self.show_progress_dialog("Classifying, please wait...")
        try:
            with open(self.uploaded_file, 'rb') as f:
                files = {'file': (self.uploaded_file, f, 'application/octet-stream')}
                params = {'classifier': self.api_classifer[classifier], "value": model_param}
                response = requests.post(api_url, files=files, params=params)
                
                if response.status_code == 200:
                    data_json = response.json()
                    result = data_json['accuracy']
                    accuracy = "{:.2f}%".format(result)
                    
                    QMessageBox.information(self, "Success", f"Model Accuracy : {accuracy}")
                    
                    if "data" in data_json:
                        new_df = pd.DataFrame(data_json["data"])
                        self.modified_data = new_df.copy()
                        self.populate_table(self.modified_data, "UnAnnotated Dataset", 0, 1)
                    self.show_confusion_matrix_window()
                    self.save_results_to_json(classifier, param_label, model_param, accuracy)
                else:
                    self.show_error(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            self.show_error(f"Failed to upload file: {str(e)}")
        finally:
            
            self.hide_progress_dialog()
        

    def show_error(self, message):
            error_dialog = QErrorMessage(self)
            error_dialog.showMessage(message)

    def upload_file_for_classification(self, api_url, file, classifier, model_param):
        try:

            with open(file, 'rb') as f:
                files = {'file': (file, f, 'application/octet-stream')}
                params = {'classifier': classifier, "value" : model_param}

                response = requests.post(api_url, files=files, params=params)
                result = response.json()['accuracy']
                accuracy = "{:.2f}".format(result)
                
                if response.status_code == 200:
                    QMessageBox.information(self, "Success", f"Model Accuracy : {accuracy}%")
                    print("File processed successfully")

                else:
                    print(f"Error: {response.status_code}, {response.text}")
                    self.show_error(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            self.show_error(f"Failed to upload file: {str(e)}")


    def show_pie_chart(self, df):

        df.iloc[:, 0] = df.iloc[:, 0].astype(str)

        sentiment_counts = df.iloc[:, 0].value_counts()
        print(sentiment_counts)

        sizes = [
            sentiment_counts.get('0', 0),
            sentiment_counts.get('2', 0),
            sentiment_counts.get('4', 0)
        ]

        labels = [
            self.sentiments.get('0', 'Unknown'),
            self.sentiments.get('2', 'Unknown'),
            self.sentiments.get('4', 'Unknown')
        ]

        self.figure.clear()

        ax = self.figure.add_subplot(111, aspect='equal')

        total = sum(sizes)

        if total == 0:

            ax.text(
                0.5, 0.5, 'No data available',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12
            )
        else:

            ax.pie(
                sizes, labels=labels, autopct='%1.1f%%',
                startangle=90, colors=["#FF5252", "#FFEB3B", "#4CAF50"]
            )
            ax.axis('equal') 


        ax.set_facecolor("none")

        self.canvas.setVisible(True)
        self.canvas.draw()
        self.position_pie_chart()
        self.show_stats_table()
  

        



    def position_pie_chart(self):

        window_rect = self.geometry()
        table_rect = self.tweet_table.geometry()

        top_padding = 120
        right_padding = 0
        left_padding = -50
        bottom_padding = 10

        pie_width = int(table_rect.width() * 0.5)
        pie_height = int(table_rect.height() * 0.5)

        available_space_bottom = window_rect.bottom() - pie_height - top_padding - bottom_padding
        available_space_right = window_rect.right() - pie_width - right_padding - left_padding

        pie_x = available_space_right
        pie_y = window_rect.top() + top_padding

        if pie_x < window_rect.left() + left_padding:
            pie_x = window_rect.left() + left_padding

        if pie_y + pie_height > window_rect.bottom() - bottom_padding:
            pie_y = window_rect.bottom() - bottom_padding - pie_height

        top_layout_height = self.top_layout.sizeHint().height()
        if pie_y < top_layout_height + top_padding:
            pie_y = top_layout_height + top_padding

       
        self.canvas.setGeometry(QRect(pie_x, pie_y, pie_width, pie_height))
        self.canvas.setVisible(True)



    def show_confusion_matrix_window(self):
        conf_matrix = np.array([[50, 10, 5], [12, 60, 3], [8, 4, 45]])
        labels = ['Positive', 'Negative', 'Neutral']
               
        self.confusion_window.show_confusion_matrix(conf_matrix, labels)
        self.confusion_window.show()

    def show_progress_dialog(self, message):
        self.progress_dialog = ProgressDialog(message, self)
        self.progress_dialog.show()
   
        
        QApplication.processEvents()

    def hide_progress_dialog(self):
        if hasattr(self, 'progress_dialog') and self.progress_dialog.isVisible():
            self.progress_dialog.close()
            del self.progress_dialog


    def save_results_to_json(self, classifier, param_label, model_param, accuracy):
 
        results_data = {
            "choose_algorithm": classifier,
            "model_parameters_label": param_label,
            "model_parameters_value": model_param,
            "result_accuracy": accuracy,
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        results_file_path = os.path.join('datasets', 'db', 'resultsdb.json')
        os.makedirs(os.path.dirname(results_file_path), exist_ok=True)

        if os.path.exists(results_file_path):
            with open(results_file_path, 'r') as rf:
                try:
                    existing_data = json.load(rf)
                    if not isinstance(existing_data, list):
                        existing_data = []
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []

        existing_data.append(results_data)

        with open(results_file_path, 'w') as wf:
            json.dump(existing_data, wf, indent=4)




    def show_stats_table(self):

        if hasattr(self, 'stats_table') and self.stats_table is not None:
            self.right_layout.removeWidget(self.stats_table)
            self.stats_table.deleteLater()
            self.stats_table = None  


        results_file_path = os.path.join('datasets', 'db', 'resultsdb.json')
        if not os.path.exists(results_file_path):
            print(f"Results file not found at {results_file_path}.")
            return

        try:
            with open(results_file_path, 'r') as rf:
                data = json.load(rf)
                if not isinstance(data, list):
                    print("JSON data is not a list.")
                    return

                data.sort(key=lambda x: x.get("training_date", ""), reverse=True)
                top_entries = data[:5]
        except Exception as e:
            print("Error reading JSON:", e)
            return

        if not top_entries:
            print("No top entries found.")
            return


        self.stats_table = QTableWidget(self)
        self.stats_table.setColumnCount(5)
        self.stats_table.setHorizontalHeaderLabels(["Algorithm", "Param Label", "Param Value", "Accuracy", "Date"])
        self.stats_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.stats_table.setRowCount(len(top_entries))


        font = self.stats_table.font()
        font.setPointSize(12)
        self.stats_table.setFont(font)
        self.stats_table.setStyleSheet("""
            QTableWidget {
                background-color: black;
                color: white;
                gridline-color: white;
            }
            QHeaderView::section {
                background-color: gray;
                color: white;
            }
        """)


        for i, entry in enumerate(top_entries):
            alg_item = QTableWidgetItem(entry.get("choose_algorithm", ""))
            param_label_item = QTableWidgetItem(entry.get("model_parameters_label", ""))
            param_value_item = QTableWidgetItem(entry.get("model_parameters_value", ""))
            accuracy_item = QTableWidgetItem(entry.get("result_accuracy", ""))
            date_item = QTableWidgetItem(entry.get("training_date", ""))


            for item in [alg_item, param_label_item, param_value_item, accuracy_item, date_item]:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setBackground(QBrush(QColor("#696969")))
                item.setForeground(QBrush(QColor("white")))


            self.stats_table.setItem(i, 0, alg_item)
            self.stats_table.setItem(i, 1, param_label_item)
            self.stats_table.setItem(i, 2, param_value_item)
            self.stats_table.setItem(i, 3, accuracy_item)
            self.stats_table.setItem(i, 4, date_item)


        self.stats_table.resizeColumnsToContents()
        self.stats_table.verticalHeader().setDefaultSectionSize(45)
        self.stats_table.horizontalHeader().setDefaultSectionSize(125)

    
        self.right_layout.addWidget(self.stats_table)
        self.stats_table.setVisible(True)


        self.position_stats_table()

        
        # self.refresh_other_elements()



    def position_stats_table(self):
        window_rect = self.geometry()
        pie_rect = self.canvas.geometry()  
        table_rect = self.tweet_table.geometry() 


        stats_width = 650
        stats_height = 250

  
        left_padding = 500  
        top_padding = 220
        right_padding = 120
        bottom_padding = 20

       
        stats_x = pie_rect.x() + left_padding
        stats_y = pie_rect.y() + pie_rect.height() + 20 + top_padding 

   
        if stats_x <= table_rect.right() + left_padding:
            stats_x = table_rect.right() + left_padding

  
        if stats_x + stats_width > window_rect.right() - right_padding:
            stats_x = window_rect.right() - stats_width - right_padding

        if stats_y + stats_height > window_rect.bottom() - bottom_padding:
            stats_y = window_rect.bottom() - stats_height - bottom_padding

        if stats_x < window_rect.left() + left_padding:
            stats_x = window_rect.left() + left_padding

        if stats_y < window_rect.top() + top_padding:
            stats_y = window_rect.top() + top_padding

        self.stats_table.setGeometry(QRect(stats_x, stats_y, stats_width, stats_height))
        self.stats_table.setVisible(True)








class ConfusionMatrixWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Confusion Matrix")
        self.setMinimumSize(600, 400)

        self.layout = QVBoxLayout(self)
        self.figure = plt.Figure(figsize=(5,4), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

    def show_confusion_matrix(self, conf_matrix, labels):
        self.figure.clear()
        ax_cm = self.figure.add_subplot(111)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax_cm)
        ax_cm.set_title("Confusion Matrix")
        self.figure.tight_layout()
        self.canvas.draw()


class ProgressDialog(QDialog):
    def __init__(self, message="Please wait...", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Processing")
        self.setFixedSize(300, 100)

        layout = QVBoxLayout(self)
        self.label = QLabel(message, self)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(0) 
        layout.addWidget(self.label)
        layout.addWidget(self.progress_bar)
        self.setLayout(layout)

        self.setWindowModality(Qt.WindowModality.ApplicationModal)

