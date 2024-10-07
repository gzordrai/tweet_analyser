import pandas as pd


def confusion_matrix(row,result):
    print(row['text'],row['id'],row['target'],result)
    if result[row['id']]:
        if result[row['id']]==row['target']:
            return True
        else:
            return False

def levenshtein_distance(t1,t2):
    t1_words=t1.split(" ")
    t2_words=str(t2).split(" ")
    total_words= len(t1_words)+len(t2_words)
    common_words = 0
    for w1 in t1_words:
        if w1 in t2_words and len(w1)>1:
            common_words+=1
    distance = (total_words-common_words)/total_words
    return distance


def knn_naive(tweet, df, k):
    result={}
    dct={}
    for ind, row in df.iterrows():
        distanct = levenshtein_distance(tweet[1],row['text'])
        dct[row['id']]=(distanct,row['text'],row['target'])
    nearest = sorted(dct.items(), key=lambda x:x[1][0])
    k_nearest = nearest[:k]
    
    label = [k_nearest[i][1][2] for i in range(len(k_nearest))]
    target = max(label,key=label.count)
    result[tweet[0]]=target
    return result


def evaluate_knn(df,k):
    
    train = df.sample(frac=0.75,random_state=200)
    test = df.drop(train.index)

    values=test['target'].value_counts()
    actual_positive=int(values[4])
    acutal_negative=int(values[2])
    acutal_neutral=int(values[0])
    total_values = actual_positive+acutal_negative+acutal_neutral
    correct=0
    error=0
    for ind,row in test.iterrows():
        result = knn_naive((row['id'],row['text']),df,k)
        if confusion_matrix(row,result):
            correct+=1
        else:
            error+=1
    print(total_values)
    print(correct,error)


if __name__=="__main__":
    df = pd.read_csv('datasets\cleaned_data.csv')
    df.columns = ['target','id','date','flag','user','text']
    evaluate_knn(df,5)
    
