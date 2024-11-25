from fastapi import FastAPI, HTTPException, UploadFile, File
from os import remove
from os.path import abspath, dirname, join
from shutil import copyfileobj
from sys import path

path.append(abspath(join(dirname(__file__), '..', '..')))

from core.classifier import KNNClassifier, NaiveBayesClassifier
from core.dataset import Dataset

app: FastAPI = FastAPI()
knn = KNNClassifier()
bayes = NaiveBayesClassifier()

@app.post("/dataset/train")
async def upload_file_for_training(file: UploadFile = File(...), classifier: str = "knn"):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Only CSV files are allowed.")
    
    # Save the uploaded file to a temporary location
    temp_file_path = f"/tmp/{file.filename}"

    with open(temp_file_path, "wb") as buffer:
        copyfileobj(file.file, buffer)
    
    result = process_file(temp_file_path)
    
    remove(temp_file_path)
    
    return {"result": result}

@app.post("/dataset/test")
async def upload_file_for_testing(file: UploadFile = File(...), classifier: str = "knn"):
    pass

def process_file(file_path: str):
    dataset = Dataset(file_path, knn)
    dataset.classify()

    return "File processed successfully"