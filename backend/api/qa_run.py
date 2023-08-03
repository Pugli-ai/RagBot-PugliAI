import pandas as pd
import openai
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
try:
    from api import variables_db
    from api import pinecone_functions
    
except:
    import variables_db
    import pinecone_functions

from openai.embeddings_utils import distances_from_embeddings
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
import traceback



def create_context(question, top_k=3, max_len=1800):
    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    results = pinecone_functions.INDEX.query(
    vector=q_embeddings, 
    top_k=3, 
    include_metadata=True)

    texts = []
    cur_len = 0

    for result in results['matches']:
        cur_len += result['metadata']['n_tokens'] + 4
        if cur_len > max_len:
            break
        texts.append(result['metadata']['text'])

    source_url = results['matches'][0]['metadata']['url']

    
    return "\n\n###\n\n".join(texts), source_url

def conversation(
    context,
    question,
    model="text-davinci-003",
    max_len=2000,
    size="ada",
    debug=False,
    max_tokens=1500,
    stop_sequence=None
):
        template = """Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"
        \n\nContext: {context}\n\n
        ---\n\nQuestion: {question}
        \nAnswer:"""

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )
        LLM = OpenAI(
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            #stop=stop_sequence,
            model=model,
            client=openai.ChatCompletion
        )
        chatgpt_chain = LLMChain(
            llm=LLM,
            prompt=prompt,
            verbose=False,
        )

        output = chatgpt_chain.predict(context=context, question=question)
        return output



def answer_question(
    model="text-davinci-003",
    question="Am I allowed to publish model outputs to Twitter, without a human review?",
    max_len=2000,
    size="ada",
    debug=False,
    max_tokens=1500,
    stop_sequence=None
):
    
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context, source_url = create_context(question)
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the question and context
        """
        response = openai.Completion.create(
            prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        # Add the source URL to the answer
        answer = response["choices"][0]["text"].strip()
        """
        conversation_output = conversation(context, question)
        answer =  conversation_output
 
        dunno_list = ["I don't know", "I don’t know", "I do not know", "I don’t know", "I don't know.", "I don’t know.", "I do not know.", "I don’t know."]
        if answer.strip() in dunno_list:
            source_url = None
            success = False
        else:
            success = True

        return {"answer": answer, "source_url": source_url, "success": success, "error_message": None}
    except Exception as e:
        error_message = traceback.format_exc().splitlines()
        error_message = [x for x in error_message if x.strip()]
        error_message = error_message[-1]
        return {"answer": "Error!", "source_url": None, "success": False, "error_message": str(error_message)}


def main(question, openai_api_key, pinecone_index_name):
    try: 
        if variables_db.OPENAI_API_KEY != openai_api_key:
            print('Changing OPENAI_API_KEY')
            variables_db.OPENAI_API_KEY = openai_api_key
            openai.api_key = variables_db.OPENAI_API_KEY

        pinecone_index_name = pinecone_functions.url_to_index_name(pinecone_index_name)
        if not pinecone_functions.is_db_exists(pinecone_index_name):
            return {"answer": "Error!", "source_url": None, "success": False, "error_message": f"There is no database with name {pinecone_index_name}"}
        if variables_db.PINECONE_INDEX_NAME != pinecone_index_name:
            print('Changing PINECONE_INDEX')
            variables_db.PINECONE_INDEX_NAME = pinecone_index_name
            pinecone_functions.INDEX = pinecone_functions.retrieve_index()
    
        response = answer_question(question=question, debug=False)
    except Exception as e :
        error_message = traceback.format_exc().splitlines()
        error_message = [x for x in error_message if x.strip()]
        error_message = error_message[-1]
        response = {"answer": "Error!", "source_url": None, "success": False, "error_message": str(error_message)}
    
    return response


def init():
    """ Initialize the app's neccessary components
    """
    # Connecting to Pinecone
    pinecone_functions.init_pinecone(variables_db.PINECONE_API_KEY, variables_db.PINECONE_API_KEY_ZONE)
    load_dotenv()
    openai.api_key = variables_db.OPENAI_API_KEY

init() # initialize the app's neccessary components


if __name__=="__main__":


    #full_url = "https://www.deghi.it/supporto/"
    full_url= "https://gethelp.tiledesk.com/"

    """
    question = "Tutti gli articoli su Deghi sono disponibili?"
    answer = main(question, variables_db.OPENAI_API_KEY, full_url)
    print(f"Question: {question}\nAnswer: {answer}")
    """ 

    question = "What is tiledesk"
    answer = main(question, variables_db.OPENAI_API_KEY, full_url)
    print(f"Question: {question}\nAnswer: {answer}")

    question = "What day is it?"
    answer = main(question, variables_db.OPENAI_API_KEY, full_url)
    print(f"Question: {question}\nAnswer: {answer}")

    question="which javascript code should i copy and paste for installing the widget on my website ? please write me that javascript code"
    answer = main(question, variables_db.OPENAI_API_KEY, full_url)
    print(f"Question: {question}\nAnswer: {answer}")

    question="How to connect Tiledesk with Telegram"
    answer = main(question, variables_db.OPENAI_API_KEY, full_url)
    print(f"Question: {question}\nAnswer: {answer}")
