from fastapi import UploadFile
import pandas as pd
import os

from ml_logic.model_operations import Ml_Model

file_directory = os.path.abspath("./datasets")

if not os.path.exists(file_directory):
    os.makedirs(file_directory)



ml = Ml_Model()

def save_file(file : UploadFile):
    file_name = file.filename
    file_location = os.path.join(file_directory,file_name)
    with open(file_location,"wb") as f:
        content = file.file.read()
        f.write(content)
    ml.read_file(file_location)
    return file_location
    