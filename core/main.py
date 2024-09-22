from fastapi import FastAPI
from api import testing_api,training_api

app=FastAPI()

app.include_router(training_api.router, prefix='/train_ml')
app.include_router(testing_api.router, prefix='/test_ml')





