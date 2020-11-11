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
        
        part= tracker.get_slot("boat_part")

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
        # sentences=['need some pills for PROBLEM1',
        #     'Prob2 can i get some mdeicine.'
        #     ,'No problem',
        #     'yeah it\'s good now',
        #     'I have issues with  prob3',
        #     'Cracking in the hull','Hul is important']

        # solutions=['MED1','MED2','no_2','no_3','MED3','Need some professional help','no_6']

        embeddings = model.encode(sentences)

        return [model,embeddings,solutions]


class ActionDemo(ActionGreet):

    def name(self) -> Text:
        return "action_demo"
    
    def run(self ,dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text,Any]) -> List[Dict[Text,Any]]: 


            model,embeddings,solutions = ActionGreet.loader(self)

            message=tracker.latest_message['text']

            test=model.encode(message)

            for i in range(len(solutions)):
                cos_sim = util.pytorch_cos_sim(test, embeddings)
                
            cos_sim=cos_sim.tolist()
            print(cos_sim)
            sol_index=cos_sim[0].index(max(cos_sim[0]))

            solution=solutions[sol_index]
            print(solution)
            dispatcher.utter_message(text=solution)

            return []

    