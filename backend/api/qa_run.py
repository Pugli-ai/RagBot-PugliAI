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

# List of possible "I don't know" responses.
DUNNO_LIST = [
    "I don't know", "I don’t know", "I do not know", "I don’t know",
    "I don't know.", "I don’t know.", "I do not know.", "I don’t know."
]

def create_context(question: str, top_k: int = 5, max_len: int = 1800) -> tuple:
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
    results = pinecone_functions.INDEX.query(vector=q_embeddings, top_k=top_k, include_metadata=True)
    
    texts = []
    cur_len = 0
    for result in results['matches']:
        cur_len += result['metadata']['n_tokens'] + 4
        if cur_len > max_len:
            break
        texts.append(result['metadata']['text'])
    
    source_url = results['matches'][0]['metadata']['url']
    
    return "\n\n###\n\n".join(texts), source_url

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
    * Your answer should be in json format as below to machine to understand.
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

def answer_question(question: str, chat_history: str = "") -> dict:
    """
    Answer a question based on the most similar context.

    Args:
        question (str): The question to be answered.
        model (str, optional): The model to be used for answering. Defaults to "text-davinci-003".
        max_tokens (int, optional): Maximum tokens for the response. Defaults to 1500.

    Returns:
        dict: A dictionary containing the answer, source URL, success status, and error message (if any).
    """

    context, source_url_old = create_context(question)
    #print("source url : ", source_url)
    #print("content : ", context)

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

def main(question: str, openai_api_key: str, pinecone_index_name: str, chat_history_dict:dict = dict()) -> dict:
    """
    Main function to answer a question based on the most relevant context from a database.

    Args:
        question (str): The question to be answered.
        openai_api_key (str): The API key for OpenAI.
        pinecone_index_name (str): The name of the Pinecone index.

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
        pinecone_index_name = pinecone_functions.url_to_index_name(pinecone_index_name)
        
        # Check if the Pinecone database exists.
        if not pinecone_functions.is_db_exists(pinecone_index_name):
            return {"answer": "Error!", "source_url": None, "success": False, "error_message": f"There is no database with name {pinecone_index_name}"}
        
        # Check and update the Pinecone index if necessary.
        if variables_db.PINECONE_INDEX_NAME != pinecone_index_name:
            print('Changing PINECONE_INDEX')
            variables_db.PINECONE_INDEX_NAME = pinecone_index_name
            pinecone_functions.INDEX = pinecone_functions.retrieve_index()
        chat_history = create_chat_history_string(chat_history_dict)
        answer = answer_question(question=question, chat_history = chat_history)
        print(f'{datetime_now} // Answer: {answer}')
        return answer
    except Exception as e:
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
    pinecone_functions.init_pinecone(variables_db.PINECONE_API_KEY, variables_db.PINECONE_API_KEY_ZONE)
    load_dotenv()
    openai.api_key = variables_db.OPENAI_API_KEY
    os.environ['OPENAI_API_KEY'] = openai.api_key
    

init()

if __name__ == "__main__":

    full_url = "https://www.deghi.it/supporto/"
    #full_url= "https://gethelp.tiledesk.com/"
    #full_url = "https://docs.pinecone.io/"

    if full_url == "https://gethelp.tiledesk.com/":

        question = "What is tiledesk?"
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