from fastapi import FastAPI, HTTPException, UploadFile, File
from os import remove
from os.path import abspath, dirname, join
from shutil import copyfileobj
from sys import path
import os
import tempfile
import pandas as pd
path.append(abspath(join(dirname(__file__), '..', '..')))

from core.classifier import KNNClassifier, NaiveBayesClassifier
from core.dataset import AnnotatedDataset, UnannotateDataset
 
app: FastAPI = FastAPI()

temp_path = '../../datasets/temp'
annota_path = '../../datasets/annotated'

class PreprocessAPI():
    def __init__(self, value):
          self._df = pd.DataFrame() 
          self.bayes = NaiveBayesClassifier(value) 
          self.knn = KNNClassifier(value)        
          self.classifiers = {"knn": self.knn , "bayes" : self.bayes }
          

    def upload_file(self, file: UploadFile, path):
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Invalid file type. Only CSV files are allowed.")
        
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv', dir=temp_dir)
        temp_file_path = temp_file.name  
        
        print(f"Temporary file path: {temp_file_path}") 
        
        with open(temp_file_path, "wb") as buffer:
            copyfileobj(file.file, buffer)
        
        return temp_file_path

    def get_accuracy(self, dataset ,classifier: str, value : int):
        try:
            if classifier=='bayes':
                    self.bayes.train(dataset.get_data())
                    accuracy = dataset.classify()
                    return accuracy
            elif classifier=='knn':
                    accuracy = dataset.classify()
                    return accuracy
        except:
            raise "Error during traning process"

    def process_file(self, file_path: str, api_type: str, classifier: str, value : int):
        if api_type == "annotated":
            dataset = AnnotatedDataset(file_path, self.classifiers[classifier])
            accuracy = self.get_accuracy(dataset, classifier, value)
            cm_df = dataset.calculate_confusion_matrix()
            return accuracy, cm_df
        elif api_type == "unannotated":
            annotateddataset = AnnotatedDataset(r'datasets\inputs\testdata.manual.2009.06.14.csv', self.classifiers[classifier])
            dataset = UnannotateDataset(file_path, self.classifiers[classifier], annotateddataset)
            accuracy = self.get_accuracy(annotateddataset, classifier, value)
            tmp, self._df = dataset.classify()
            return accuracy, self._df
        else:
            raise HTTPException(status_code=400, detail="Invalid API call type")

@app.post("/dataset/annotated")
async def upload_file_for_training(file: UploadFile = File(...), classifier: str = "knn", value : int = 1):
    api = PreprocessAPI(value)
    temp_file_path = api.upload_file(file, temp_path)
    result, cm_df = api.process_file(temp_file_path, "annotated", classifier, value)
    data = cm_df.to_dict(orient='records')
    return {"accuracy": result, "cm": data}

@app.post("/dataset/unannotated")
async def upload_file_for_testing(file: UploadFile = File(...), classifier: str = "knn" , value : int = 1):
    api = PreprocessAPI(value)
    temp_file_path = api.upload_file(file, temp_path)
    result, df = api.process_file(temp_file_path, "unannotated", classifier, value)
    data = df.to_dict(orient='records')
    
    return {"accuracy": result, "data" : data}