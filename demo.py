from sentence_transformers import SentenceTransformer, util
import warnings

import pandas as pd
warnings.filterwarnings("ignore")
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

test=str(input("INPUT :-"))

df=pd.read_csv('/home/sahib/Downloads/faq-rasa.csv')
sentences=df['Questions'].str.replace("\n", "", case = False).tolist()
solutions=df['Answers'].str.replace("\n", "", case = False).tolist()
#Encode all sentences
def input_sent(sentences):
    embeddings = model.encode(sentences)
    return embeddings

embeddings=input_sent(sentences)

def encode(test):
    test=model.encode(test)
    return test
test=encode(test)

cos_sim = util.pytorch_cos_sim(test, embeddings)
cos_sim=cos_sim.tolist()




sol_index=cos_sim[0].index(max(cos_sim[0]))

print('SOLUTION IS :-',solutions[sol_index])