from csv import reader
from os import getcwd
from os.path import join
from re import sub
import pandas as pd

def clean_data(data: str) -> str:
    patterns = [
        (r"@[a-zA-Z0-9]+", ""),                                                                         # Remove mentions
        (r"#[a-zA-Z0-9]+", ""),                                                                         # Remove hashtags
        (r"RT", ""),                                                                                    # Remove retweets
        (r".+ - http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", ""), # Remove attached links
        (r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", ""),      # Remove links
        (r".*[:;][\)][^\n]*[:;][\(].*|.*[:;][\(][^\n]*[:;][\)].*", ""),                                 # Remove happy and sad emoticons in the same tweet
        (r"(?<=[a-zA-Z])[!\?\"\.;,]", r" \g<0>"),                                                       # Add space before punctuation only if there's a letter before
        (r"", "")                                                                                       # Remove any remaining whitespace
    ]

    for pattern in patterns:
        if data == "":
            break

        data = sub(pattern[0], pattern[1], data)

    return data

def pre_processing():
    with open(join(getcwd(), r"datasets\testdata.manual.2009.06.14.csv"), "r") as f, open(join(getcwd(), "datasets/cleaned_data.csv"), "w") as o:
        cleaned_data: set = set()
    
        for row in reader(f):
            data: str = clean_data(row[5])
    
            if data in cleaned_data:
                continue
    
            cleaned_data.add(data)
    
            if data != "":
                o.write(','.join(map(lambda x : f"\"{x}\"", row[:5])) + ',\"' + data.replace("\"", "\"\"") + "\"\n")
        

if __name__ == "__main__":
    pre_processing()
    negative=[]
    positive=[]
    with open(r"datasets\negative.txt") as neg, open(r"datasets\positive.txt") as pos:
        for line in neg:
            negative.append(set(line.split(',')))
        for line in pos:
            positive.append(set(line.split(',')))

    negative = list(negative[0])+list(negative[1])
    positive = list(positive[0])+list(positive[1])

    df = pd.read_csv('datasets\cleaned_data.csv')
    df.columns = ['target','id','date','flag','user','text']
    pos_dict={}
    neg_dict={}
    nut_dict={}
    for ind,row in df.iterrows():
        text=row['text']
        neg=set()
        pos=set()

        for word in positive:
            word=word.strip()
            if word in text and len(word)>1:
                pos.add(word)

        for word in negative:
            word = word.strip()
            if word in text and len(word)>1:
                neg.add(word)
                
        if len(pos)>len(neg):
            pos_dict[row['id']]=[(pos,len(pos))]
        elif len(pos)<len(neg):
            neg_dict[row['id']]=[(neg,len(neg))]
        else:
            nut_dict[row['id']]=[]


    # # Checking Positive Accuracy
    values=df['target'].value_counts()
    actual_positive=int(values[4])
    acutal_negative=int(values[2])
    acutal_neutral=int(values[0])
    ngng=0
    ngne=0
    ngpo=0
    neng=0
    nene=0
    nepo=0
    pong=0
    pone=0
    popo=0

    
    for ind,row in df.iterrows():
        if row["target"]==4:
            if row['id'] in pos_dict.keys():
                popo+=1
            elif row['id'] in neg_dict.keys():
                pong+=1
            elif row['id'] in nut_dict.keys():
                pone+=1

        if row["target"]==2:
            if row['id'] in pos_dict.keys():
                ngpo+=1
            elif row['id'] in neg_dict.keys():
                ngng+=1
            elif row['id'] in nut_dict.keys():
                ngne+=1

        if row["target"]==0:
            if row['id'] in nut_dict.keys():
                nene+=1
            elif row['id'] in pos_dict.keys():
                nepo+=1
            elif row['id'] in neg_dict.keys():
                neng+=1

    

    print("negative negative:",ngng)
    print("negative neutral:",ngne)
    print("negative positive :",ngpo)

    print("positive positive :",popo)
    print("positive negative:",pong)
    print("positive neutral:",pone)

    print("neutral neutral :",nene)
    print("neutral positive :",nepo)
    print("neutral negative:",neng)
  
    accuracy = (ngng+popo+nene)/(ngng+ngpo+ngne+popo+pong+pone+nene+neng+nepo)

    print("total accuracy :",accuracy)

    print(ngng+ngpo+ngne+popo+pong+pone+nene+neng+nepo)





    

            
    