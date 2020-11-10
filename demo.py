from sentence_transformers import SentenceTransformer, util
import pandas as pd

model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
test=str(input())

# sentences=['Now my boat is working fine',
# 'I have no issues with my boat now.'
# ,'each part of my boat is working fine',
# 'yeah it\'s good now',
# 'I have issues with  my hull',
# 'Cracking in the hull','Hull is integral part of boat']
df=pd.read_csv('/home/sahib/Downloads/faq-rasa.csv')
sentences=df['Questions'].str.replace("\n", "", case = False).tolist()
solutions=df['Answers'].str.replace("\n", "", case = False).tolist()

#Encode all sentences
embeddings = model.encode(sentences)
test=model.encode(test)
for i in range(len(sentences)):
    cos_sim = util.pytorch_cos_sim(test, embeddings)
cos_sim=cos_sim.tolist()

print('MAXIMUM SIMILARITY IS :- ',cos_sim.index(max(cos_sim)))