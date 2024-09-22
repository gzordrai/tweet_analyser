from fastapi import APIRouter

router = APIRouter()

@router.get('/')
def hello_pje():
    return {"Hello":"X Sentimantial Analysis"}