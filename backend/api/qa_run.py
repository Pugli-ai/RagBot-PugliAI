import os
import openai
from dotenv import load_dotenv
import traceback
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI

# Attempt to import necessary modules from the API, if not available, import them directly.
try:
    from api import variables_db
    from api import pinecone_functions
except ImportError:
    import variables_db
    import pinecone_functions
import json
from datetime import datetime

########################################################### CHILD FUNCTIONS ###########################################################
#######################################################################################################################################

def create_context(question: str, pinecone_namespace: str, top_k: int = 5, max_len: int = 1800) -> tuple:
    """
    Create a context for the given question using embeddings.

    Args:
        question (str): The question for which context is needed.
        top_k (int, optional): Number of top matches to consider. Defaults to 3.
        max_len (int, optional): Maximum length of the context. Defaults to 1800.

    Returns:
        tuple: A tuple containing the context and the source URL.
    """
    # Generate embeddings for the question.
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']
    
    # Query the pinecone index to get the most relevant matches.
    results = pinecone_functions.INDEX.query(
        namespace = pinecone_namespace,
        vector=q_embeddings,
        top_k=top_k,
        include_metadata=True)
    
    texts = []
    cur_len = 0
    for result in results['matches']:
        cur_len += result['metadata']['n_tokens'] + 4
        if cur_len > max_len:
            break
        texts.append(result['metadata']['text'])
    
    #source_url = results['matches'][0]['metadata']['url']
    
    return "\n\n###\n\n".join(texts)

def conversation(context: str, question: str, chat_history: str = "", model: str = "gpt-4", max_tokens: int = 1500) -> str:
    """
    Generate a conversation based on the provided context and question.

    Args:
        context (str): The context for the conversation.
        question (str): The question to be answered.
        chat_history (str)
        model (str, optional): The model to be used for the conversation. Defaults to "gpt-4".
        max_tokens (int, optional): Maximum tokens for the response. Defaults to 1500.

    Returns:
        str: The model's response.
    """
    
    template = f"""
    Answer the question based on the context below. If the answer isn't in the context, refer to the chat history. If the answer can't be found in either, say "I don't know".

    ---

    Context: {context}

    ---

    Chat History:
    {chat_history}

    ---

    Question: {question}
    Answer:
    """

    response = openai.ChatCompletion.create(
        messages=[
            {
            "role": "user",
            "content": template
            }
        ],
        #prompt=template,
        temperature=0,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        model=model,
    )

    return response['choices'][0]['message']['content']

def convert_question_to_english(question:str):

    prompt = f"""
                Convert below text to English, if it is already in English, return the same text. But provide it in a json format like below. The response should be json format otherwise it will not work.
                {{
                    "original_text": {question},
                    "translated_text": "text", 
                    "original_language": "language"
                }}
            """
    

    response = openai.ChatCompletion.create(
        messages=[
            {
            "role": "user",
            "content": prompt
            }
        ],
        #prompt=template,
        temperature=0,
        max_tokens=1500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        model="gpt-4",
    )

    return json.loads(response['choices'][0]['message']['content'])

def conversation_with_langchain(context: str, question: str, chat_history: str = "", model: str = "gpt-4", max_tokens: int = 1500) -> str:
    """
    Generate a conversation based on the provided context and question.

    Args:
        context (str): The context for the conversation.
        question (str): The question to be answered.
        model (str, optional): The model to be used for the conversation. Defaults to "gpt-4".
        max_tokens (int, optional): Maximum tokens for the response. Defaults to 1500.

    Returns:
        str: The model's response.
    """
    # Define the template for the conversation.
    #translation = convert_question_to_english(question)
    #question = translation['translated_text']
    #print("question : ", question)
    #language = translation['original_language']
    template = """
    You are a chatbot seeking precise answers to given questions by exploring a webpage and its subpages. Your goal is to sift through the provided context to find the most accurate answer to the question asked. If the context contains a direct or related answer, provide it with question's language. If the answer is not present in the context or chat history say "I don't know".
    These are the rules;
    * Always make a respond to greetings messages you can say like "hi how can i help you?".
    * Translate your answer to the question's language.
    * Format your response as a JSON object. This is too important otherwise it will not work. The root key should be "Response".
    * If the answer in the context provide its source url. You will find it in the context. Otherwise make source url None.
    ---

    Context: {context}

    ---

    Chat History:
    {chat_history}

    ---

    Question: {question}

    {{"Response" :
        {{
        "questions_language" : "language",
        "answer": "answer in question's language",
        "source_url": "source_url"
        }}
    }}
    """
    
    prompt = PromptTemplate(input_variables=["context", "chat_history", "question"], template=template)
    LLM = ChatOpenAI(temperature=0, max_tokens=max_tokens, model=model)
    chatgpt_chain = LLMChain(llm=LLM, prompt=prompt, verbose=False)
    answer = chatgpt_chain.predict(context=context, chat_history=chat_history, question=question)
    answer_json = json.loads(answer)
    
    return answer_json['Response']

def handle_exception(e: Exception) -> dict:
    """
    Handle exceptions and return a formatted error message.

    Args:
        e (Exception): The exception to be handled.

    Returns:
        dict: A dictionary containing the error message.
    """
    error_message = traceback.format_exc().splitlines()
    error_message = [x for x in error_message if x.strip()]
    
    return {"answer": "Error!", "source_url": None, "success": False, "error_message": error_message[-1]}

def answer_question(question: str, pinecone_namespace: str, chat_history: str = "") -> dict:
    """
    Answer a question based on the most similar context.

    Args:
        question (str): The question to be answered.
        model (str, optional): The model to be used for answering. Defaults to "text-davinci-003".
        max_tokens (int, optional): Maximum tokens for the response. Defaults to 1500.

    Returns:
        dict: A dictionary containing the answer, source URL, success status, and error message (if any).
    """

    context = create_context(question, pinecone_namespace)
    #print("source url : ", source_url)
    #print("########################################################")
    #print("content : ", context)
    #print("########################################################")

    try:
        answer_json = conversation_with_langchain(context, question, chat_history = chat_history)
        answer = answer_json['answer']
        source_url = answer_json['source_url']
        #print("Answer: ", answer)
        source_url = None if source_url == "None" else source_url
        success = True if source_url else False
        return {"answer": answer, "source_url": source_url, "success": success, "error_message": None}

    except Exception as e:
        traceback.print_exc()
        return handle_exception(e)

def create_chat_history_string(chat_history_dict: dict = dict()) -> str:
    """_summary_

    Args:
        chat_history_dict (dict, optional): _description_. Defaults to dict().

    Returns:
        str: _description_
    """
    chat_history= ""
    for key, value in chat_history_dict.items():
        question = value['question']
        answer = value['answer']
        chat_history+=f"\nquestion_{key}: {question}\nanswer_{key}: {answer}\n"

    return chat_history

def init() -> None:
    """
    Initialize the necessary components for the application.
    """
    #pinecone_functions.init_pinecone(variables_db.PINECONE_API_KEY, variables_db.PINECONE_API_KEY_ZONE)
    load_dotenv()
    openai.api_key = variables_db.OPENAI_API_KEY
    os.environ['OPENAI_API_KEY'] = openai.api_key
    
############################################################ MAIN FUNCTION ############################################################
#######################################################################################################################################
def main(question: str, openai_api_key: str, full_url: str, chat_history_dict:dict = dict()) -> dict:
    """
    Main function to answer a question based on the most relevant context from a database.

    Args:
        question (str): The question to be answered.
        openai_api_key (str): The API key for OpenAI.
        full_url (str): The namespace of the website on Pinecone database.
        chat_history_dict(dict): The chat history of the conversation.

    Returns:
        dict: A dictionary containing the answer, source URL, success status, and error message (if any).
    """
    try:
        #print datetime with date
        datetime_now =datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print(f'{datetime_now} // Question: {question}')
        
        # Check and update the OpenAI API key if necessary.
        if variables_db.OPENAI_API_KEY != openai_api_key:
            print('Changing OPENAI_API_KEY')
            variables_db.OPENAI_API_KEY = openai_api_key
            openai.api_key = variables_db.OPENAI_API_KEY
            os.environ['OPENAI_API_KEY'] = openai.api_key
        
        # Convert the URL to a Pinecone index name.
        pinecone_namespace = pinecone_functions.url_to_namespace(full_url)

        if pinecone_functions.is_db_exists():
            # Check and update the Pinecone index if necessary.
            pinecone_functions.INDEX = pinecone_functions.retrieve_index()
            index_stats = pinecone_functions.INDEX.describe_index_stats()
            namespaces = list(index_stats['namespaces'].keys())
            if pinecone_namespace in namespaces:
                chat_history = create_chat_history_string(chat_history_dict)
                response = answer_question(question=question, pinecone_namespace=pinecone_namespace, chat_history = chat_history)
            else:
                response = {"answer": "Error!", "source_url": None, "success": False, "error_message": f"The pinecone database found but there is no namespace called {pinecone_namespace}, please start the scraper for {full_url}"}
        else:
            response = {"answer": "Error!", "source_url": None, "success": False, "error_message": f"The pinecone database is not created, please start the scraper for {full_url}"}          

        print(f"{datetime_now} : Response: ", response)
        return response


    except Exception as e:
        traceback.print_exc()
        error_message = traceback.format_exc()
        return {"answer": "Error!", "source_url": None, "success": False, "error_message": error_message}          

init()

if __name__ == "__main__":

    #full_url = "https://www.deghi.it/supporto/"
    #full_url= "https://gethelp.tiledesk.com/"
    #full_url = "https://docs.pinecone.io/"

    #full_url = "https://knowledge.webafrica.co.za"
    full_url = "https://aulab.it/"
    #full_url = "https://ecobaby.it/"
    #full_url = "https://lineaamica.gov.it/" # TRY IT LONGER
    #full_url = "http://cairorcsmedia.it/"
    #full_url = "https://www.sace.it/"
    #full_url = "https://www.ucl.ac.uk/"
    #full_url = "http://iostudiocongeco.it/"
    #full_url = "https://www.postpickr.com/"
    #full_url = "https://www.loloballerina.com/"

    if full_url == "https://gethelp.tiledesk.com/":
        question_list = [
                        "What is tiledesk?",
                        "What day is it?",
                        "which javascript code should i copy and paste for installing the widget on my website ? please write me that javascript code",
                        "Ciao",
                        "How to connect Tiledesk with Telegram",

        ]
        for question in question_list:
            answer = main(question, variables_db.OPENAI_API_KEY, full_url)

    elif full_url == "https://www.deghi.it/supporto/":
        question_list = [
                        "Quanto ci mette in genere un pacco ad arrivare?",
                        "How long does a package usually take to arrive?",
                        "How are you?",
                        "Ciao",
                        "superati i 14 giorni posso effettuare il diritto di recesso?",
                        "quanto tempo ho a disposiione per un reso",
                        "posso pagare con binifico bancario",
                        "come posso pagare",
                        "tempo di attesa per una spedizione",
                        "quanto tempo devo aspettare per un ordine?",
                        "quanto tempo ci vuole per ricevere un ordine?"
        ]
        for question in question_list:
            answer = main(question, variables_db.OPENAI_API_KEY, full_url)


    elif full_url == "https://docs.pinecone.io/":
        question = "What is pinecone?"
        answer = main(question, variables_db.OPENAI_API_KEY, full_url)
        print(f"Question: {question}\nAnswer: {answer}")

        question = "How to create an index?"
        answer = main(question, variables_db.OPENAI_API_KEY, full_url)
        print(f"Question: {question}\nAnswer: {answer}")
    elif full_url == "https://knowledge.webafrica.co.za":
        question_list = [
                        "Why should i use webafrica?",
                        
        ]
        for question in question_list:
            answer = main(question, variables_db.OPENAI_API_KEY, full_url)        
    elif full_url == "https://ecobaby.it/":
        question_list = [
                        "What is ecobaby?",
                        
        ]
        for question in question_list:
            answer = main(question, variables_db.OPENAI_API_KEY, full_url)
    elif full_url == "https://aulab.it/":
        question_list = [
                        "What is web programming?",
                      
        ] 
        for question in question_list:
            answer = main(question, variables_db.OPENAI_API_KEY, full_url)
    elif full_url == "https://lineaamica.gov.it/":
        question_list = [
                        "What is web lineaamicia?",
                      
        ] 
        for question in question_list:
            answer = main(question, variables_db.OPENAI_API_KEY, full_url)

    elif full_url == "http://cairorcsmedia.it/":
        question_list = [
                        "What is cairorcs media?",
                      
        ] 
        for question in question_list:
            answer = main(question, variables_db.OPENAI_API_KEY, full_url)

    elif full_url == "https://www.sace.it/":
        question_list = [
                        "What is sace?",
                      
        ] 
        for question in question_list:
            answer = main(question, variables_db.OPENAI_API_KEY, full_url)

    elif full_url == "https://www.ucl.ac.uk/":
        question_list = [
                        "What is ucl?",
                      
        ] 
        for question in question_list:
            answer = main(question, variables_db.OPENAI_API_KEY, full_url)

    elif full_url == "http://iostudiocongeco.it/":
        question_list = [
                        "What is iostudiocongeco?",
                      
        ] 
        for question in question_list:
            answer = main(question, variables_db.OPENAI_API_KEY, full_url)

    elif full_url == "https://www.postpickr.com/":
        question_list = [
                        "What is postpickr?",
                      
        ] 
        for question in question_list:
            answer = main(question, variables_db.OPENAI_API_KEY, full_url)

    elif full_url == "https://www.loloballerina.com/":
        question_list = [
                        "What is loloballerina?",
                      
        ] 
        for question in question_list:
            answer = main(question, variables_db.OPENAI_API_KEY, full_url)
'''
Prompts,
template = """
You are an individual seeking precise answers to your questions by exploring a webpage and its subpages. Your goal is to sift through the provided context to find the most accurate answer to the question asked. If the context contains a direct answer, provide it. If the context contains a related but not exact answer, indicate that. If the answer is not present in the context or chat history, simply state "I don't know". Do not make guesses or generate unrelated responses. Respond in the same language as the question asked.

---

Context: {context}

---

Chat History:
{chat_history}

---

Question: {question}
Answer:
"""





template = """
Answer the question based on the context below. If the answer isn't in the context, refer to the chat history. If the answer can't be found in either, say "I don't know".

---

Context: {context}

---

Chat History:
{chat_history}

---

Question: {question}
Answer:
"""



    template = """
    You are an individual seeking precise answers to your questions by exploring a webpage and its subpages. Your goal is to sift through the provided context to find the most accurate answer to the question asked. Also you are asking with your mother language, for example if the questions is in English than your mother language is English but you are able to understand all the languages. If the context contains a direct or related answer, provide it with your mother language. If the answer is not present in the context or chat history, say exactly "I don't know" noting else. Do not make guesses or generate unrelated responses.

    ---

    Context: {context}

    ---

    Chat History:
    {chat_history}

    ---

    Question: {question}
    Answer:
    """
'''