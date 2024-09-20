from fastapi import FastAPI, File, UploadFile,Body
import os

app=FastAPI()

file_directory = os.path.abspath("./datasets")
if not os.path.exists(file_directory):
    os.makedirs(file_directory)


@app.get('/')
def read_root():
    return {"hello":"world"}


@app.post('/upload_file/')
def upload_file(file: UploadFile = File(...)):
    try:
        file_name = file.filename
        file_location = os.path.join(file_directory,file_name)
        with open(file_location,"wb") as f:
            content = file.file.read()
            f.write(content)
        return {"Upload Status":file_location}
    except Exception as  e:
        return {"Failure": str(e)}





