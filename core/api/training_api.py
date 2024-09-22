from fastapi import APIRouter, File, UploadFile
from ml_logic.utilities import save_file


router = APIRouter()

@router.post('/')
def upload_file(file: UploadFile = File(...)):
    try:
        file_location = save_file(file)
        return {"Upload Status": f"File is Uploaded to {file_location}"}
    except Exception as  e:
        return {"Failure": str(e)}





