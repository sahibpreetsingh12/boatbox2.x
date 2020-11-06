from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

print(type(model))
test=str(input("INPUT :-"))

sentences=['need some pills for PROBLEM1',
            'Prob2 can i get some mdeicine.'
            ,'No problem',
            'yeah it\'s good now',
            'I have issues with  prob3',
            'Cracking in the hull','Hul is important']

solutions=['MED1','MED2','no_2','no_3','MED3','Need some professional help','no_6']

#Encode all sentences
def input_sent(sentences):
    embeddings = model.encode(sentences)
    return embeddings

embeddings=input_sent(sentences)

def encode(test):
    test=model.encode(test)
    return test
test=encode(test)
for i in range(len(sentences)):
    cos_sim = util.pytorch_cos_sim(test, embeddings)
cos_sim=cos_sim.tolist()

print(cos_sim)

# print(max(cos_sim[0]),'\n',cos_sim)
sol_index=cos_sim[0].index(max(cos_sim[0]))

print('SOLUTION IS :-',solutions[sol_index])