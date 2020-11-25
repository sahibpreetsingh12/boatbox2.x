# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import pandas as pd


from sentence_transformers import SentenceTransformer, util
class ActionRepair(Action):

    def name(self) -> Text:
        return "action_repair"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        parts= tracker.get_slot("boat_part")

        part=parts[0] # we have extracted a list so extracting just the first entity
        if part=='hull':
            print("yes")
            dispatcher.utter_message(text="""There are two scenarios where a %s can be damaged – one is when the
      hull is damaged above the waterline, the second when it is damaged below the
      waterline.For first scenario take it out and dry it thoroughly.For hull repair,
      a basic fibreglass repair kit is used, using which the damaged section is removed
      in a circular cut. The part can be then patched using either fibreglass and
      the proper adhesives or the putties available."""%(part))
        elif part=='core':
            dispatcher.utter_message(text="""%s damage needs a professional help I would say pls visit a boat 
            repair shop near you"""%(part))

        return []



class ActionGreet(Action):

    def name(self) -> Text:
        return "action_greet"
    
    def run(self ,dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text,Any]) -> List[Dict[Text,Any]]:

            dispatcher.utter_message(text="Hi how can i help you with your boat")

            return []

    def loader(self):
        model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
        df=pd.read_csv('/home/sahib/Downloads/faq-rasa.csv')
        sentences=df['Questions'].str.replace("\n", "", case = False).tolist()
        solutions=df['Answers'].str.replace("\n", "", case = False).tolist()

        embeddings = model.encode(sentences)

        return [model,embeddings,solutions]


class ActionFAQ(ActionGreet):

    def name(self) -> Text:
        return "action_faq"
    
    def run(self ,dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text,Any]) -> List[Dict[Text,Any]]: 


            model,embeddings,solutions = ActionGreet.loader(self)

            message=tracker.latest_message['text']

            test=model.encode(message)

            parts = tracker.get_slot("boat_part") # this is a list slot so we will store all entities 
                                                    # extracted from an intent

            # to search all the strings in a list wthether in column or not                                     
            # https://stackoverflow.com/questions/17972938/check-if-a-string-in-a-pandas-dataframe-column-is-in-a-list-of-strings
            pattern = '|'.join(parts)

            sols_temp=pd.DataFrame(solutions) # converting our series to datatframe
             
            sols_temp.rename(columns = {0:'solutions'},  inplace = True)  # renaming the column

           
            if len(parts)!=0: # if no entity is extracted from user intent
                

                checker= sols_temp[sols_temp['solutions'].str.contains(pattern,na=False)]  # checking if the extracted entity
                # is present or not and if yes in which answers
                
                checker_index=list(checker.index.values) # storing the indexes of rows that were having our entity

                sols_temp=sols_temp.iloc[checker_index] # now storing only those solutions that are shortlisted

                if len(sols_temp) !=0: # if no solution is found after cosine similarity that can be because some 
                    # entities will not be present in your solution set

                    sols_temp.reset_index(level=0, inplace=True) # setting indexes again to normal

                    sols_temp.drop(columns=['index'],inplace=True) # dropping that unnecessary index column

                    emb= [embeddings[i] for i in checker_index] # stroing list of only those embeddings of questions 
                    # whose corresponding answer had the entity

                    cos_sim = util.pytorch_cos_sim(test, emb) #  cosine similarity
                        
                    cos_sim=cos_sim.tolist()
                    
                    sol_index=cos_sim[0].index(max(cos_sim[0])) # to get the index of maximum cosine similarity

                    # # p=pd.DataFrame(list(zip(cos_sim,solutions)),columns=['similarity','solutions'])
                    solution=sols_temp.iloc[[sol_index]]['solutions'][0]

                    dispatcher.utter_message(text=solution)
                    return []
                else:
                    dispatcher.utter_message(text="Sorry  But can you Rephrase it again")

                    return []


            else:
                dispatcher.utter_message(text="""Hey Really sorry but I couldn't find a Perfect Solution for
                your query. But you can rephrase and Try It Again :) """)

                return []

    
