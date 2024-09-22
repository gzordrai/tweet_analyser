import pandas as pd




class Ml_Model():
    def __init__(self):
        self.msg = "Training DataFrame"

    def read_file(self, file_path):
        df = pd.read_csv(file_path)
        print(self.msg,df.head(5))

        return df
        
    