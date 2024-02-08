import os
import openai
from dotenv import load_dotenv
import traceback
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI

try:
    from api import variables_db
    from api import pinecone_functions
except ImportError:
    import variables_db
    import pinecone_functions
import json
from datetime import datetime
import time

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

import tiktoken

context_print_option= False
max_tokens = 4097 - 317 -100 # 4097 max token size for gpt 3.5 -317 for pre prompt -100 for question
########################################################### CHILD FUNCTIONS ###########################################################
#######################################################################################################################################
def compute_token_size(text: str) -> int:
    """
    Compute the number of tokens in a string using OpenAI's tokenizer.

    Args:
        text (str): The text to be tokenized.

    Returns:
        int: The number of tokens in the text.
    """
    # Ensure that you have set your OpenAI API key
    # openai.api_key = 'your-api-key'

    # Use the GPT tokenizer to tokenize the text
    tokenizer = tiktoken.get_encoding("cl100k_base")
    return len(tokenizer.encode(text))

def truncate_text(text, max_length):
    """
    Truncate the text to the specified max_length.
    
    Args:
        text (str): The text to be truncated.
        max_length (int): The maximum length of the text.

    Returns:
        str: The truncated text.
    """
    # Split the text into words and rebuild it up to the max_length
    words = text.split()
    truncated_text = ""
    total_length = 0
    for word in words:
        total_length += len(word) + 1  # +1 for space
        if total_length > max_length:
            break
        truncated_text += word + " "
    return truncated_text.strip()

def create_context(question: str, pinecone_namespace: str, top_k: int = 5) -> tuple:
    """
    Create a context for the given question using embeddings.

    Args:
        question (str): The question for which context is needed.
        top_k (int, optional): Number of top matches to consider. Defaults to 5.
    Returns:
        tuple: A tuple containing the context and the source.
    """
    global max_tokens
    # Generate embeddings for the question.
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']
    
    # Query the pinecone index to get the most relevant matches.
    results = pinecone_functions.INDEX.query(
        namespace = pinecone_namespace,
        vector=q_embeddings,
        top_k=top_k,
        include_metadata=True)
    resource_id = results['matches'][0]['id']
    texts = []
    cur_len = 0
    for result in results['matches']:
        text = result['metadata']['content']
        n_tokens = result['metadata']['n_tokens']
        # Truncate the text if it exceeds max_tokens
        if n_tokens > max_tokens:
            text = truncate_text(text, max_tokens-4)
            n_tokens = max_tokens-4
        cur_len += n_tokens + 4  # +4 for separators
        if cur_len > max_tokens:
            break
        texts.append(text)
    
    return "\n\n###\n\n".join(texts), resource_id

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


def conversation_with_langchain(context: str, question: str, model: str, chat_history: str = "") -> str:
    """
    Generate a conversation based on the provided context and question.

    Args:
        context (str): The context for the conversation.
        question (str): The question to be answered.
        model (str, optional): The model to be used for the conversation. Defaults to "gpt-4". or "gpt-3.5-turbo"
        max_tokens (int, optional): Maximum tokens for the response. Defaults to 1500.

    Returns:
        dict: The model's response, as dictionary.
    """

    global max_tokens

    template_string = """# Chatbot Instructions
    
    You are a chatbot programmed to provide precise answers to questions by exploring a given webpage and its subpages. Your objective is to parse through the provided context to find the most accurate answer to the question asked. Follow these guidelines to ensure correct behavior:

    ## Guidelines
    - **Greeting**: Always respond to greeting messages with something like, 'Hi, how can I help you?'
    - **Source**: If the answer is found in the context, set its source. If context is not related with question say I don't know and set `source` to `None`.
    - **Answer**: Your only knowledgebase is the context and chat history. If the answer is not present in the context or chat history say "I don't know".
    - **Language**: Ensure that your answer is in the same language as the question. Always translate your answer to the question's language.

    ---

    Context: ```{context}```

    ---

    Chat History: ```{chat_history}```

    ---

    Question: ```{question}```

    format_instructions : ```{format_instructions}```
    """
    context_n_token = compute_token_size(context)
    LLM = ChatOpenAI(temperature=0, max_tokens=max_tokens-context_n_token, model=model, verbose=True)
    # Define the response schemas for structured output
    response_schemas = [
        ResponseSchema(name="questions_language", description="The language of the question."),
        ResponseSchema(name="answer", description="The answer to the question."),
        ResponseSchema(name="source", description="The source of the answer.")
    ]
    
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_template(template=template_string)
    messages = prompt.format_messages(
        context=context,
        chat_history=chat_history,
        question=question,
        format_instructions=format_instructions)
    prompt_token_size = compute_token_size(messages[0].content)
    print("FULL PROMPT TOKEN SIZE : ", prompt_token_size)
    response = LLM(messages)

    response_string = response.content.replace("\n", "")
    parsed_response = output_parser.parse(response_string)
    return parsed_response , prompt_token_size



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
    return {"answer": "Error!", "source": None, "namespace":"", "id": "", "prompt_token_size": "0", "success": False, "error_message": error_message[-1]}

def answer_question(question: str, pinecone_namespace: str, model: str, chat_history: str = "") -> dict:
    """
    Answer a question based on the most similar context.

    Args:
        question (str): The question to be answered.
        model (str, optional): The model to be used for answering. Defaults to "text-davinci-003".
        max_tokens (int, optional): Maximum tokens for the response. Defaults to 1500.

    Returns:
        dict: A dictionary containing the answer, source, success status, and error message (if any).
    """
    global context_print_option
    context_time_start = time.time()
    context, resource_id = create_context(question, pinecone_namespace)
    print("Context token size: ", compute_token_size(context))
    context_time_end = time.time()
    print("context_create_time : ", context_time_end - context_time_start)
    if context_print_option:
        print("########################################################")
        print("content : ", context)
        print("########################################################")

    try:
        conversation_time_start = time.time()
        answer_json, prompt_token_size= conversation_with_langchain(context, question, model = model, chat_history = chat_history)
        conversation_time_end = time.time()
        print("conversation_time : ", conversation_time_end - conversation_time_start)
        answer = answer_json['answer']
        source = answer_json['source']
        if source:
            source = None if source == "None" or variables_db.eof_index in source else source
        else:
            source = None

        success = True if source else False
        return {"answer": answer, "source": source, "namespace": pinecone_namespace, "id": resource_id, "prompt_token_size": prompt_token_size, "success": success, "error_message": None}

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
def main(question: str, openai_api_key: str, namespace: str, model: str, chat_history_dict:dict = dict()) -> dict:
    """
    Main function to answer a question based on the most relevant context from a database.

    Args:
        question (str): The question to be answered.
        openai_api_key (str): The API key for OpenAI.
        namespace (str): The namespace of the website on Pinecone database.
        chat_history_dict(dict): The chat history of the conversation.

    Returns:
        dict: A dictionary containing the answer, source, success status, and error message (if any).
    """
    try:
        #print datetime with date
        rome_time = pinecone_functions.get_rome_time()
        print(f'{rome_time} // Question: {question}')
        
        # Check and update the OpenAI API key if necessary.
        if variables_db.OPENAI_API_KEY != openai_api_key:
            print('Changing OPENAI_API_KEY')
            variables_db.OPENAI_API_KEY = openai_api_key
            openai.api_key = variables_db.OPENAI_API_KEY
            os.environ['OPENAI_API_KEY'] = openai.api_key
        
        # Convert the URL to a Pinecone index name.
        pinecone_namespace = namespace

        if pinecone_functions.is_db_exists():
            # Check and update the Pinecone index if necessary.
            pinecone_functions.INDEX = pinecone_functions.retrieve_index()
            index_stats = pinecone_functions.INDEX.describe_index_stats()
            namespaces = list(index_stats['namespaces'].keys())
            if pinecone_namespace in namespaces:
                chat_history = create_chat_history_string(chat_history_dict)
                response = answer_question(question=question, pinecone_namespace=pinecone_namespace, model=model, chat_history = chat_history)
            else:
                response = {"answer": "Error!", "source": None, "namespace": pinecone_namespace, "id": "", "prompt_token_size": "0", "success": False, "error_message": f"The pinecone database found but there is no namespace called {pinecone_namespace}, please start the scraper for {pinecone_namespace}"}
        else:
            response = {"answer": "Error!", "source": None, "namespace": pinecone_namespace, "id": "", "prompt_token_size": "0", "success": False, "error_message": f"The pinecone database is not created, please start the scraper for {pinecone_namespace}"}          

        print(f"{rome_time} : Response: ", response)
        return response


    except Exception as e:
        traceback.print_exc()
        error_message = traceback.format_exc()
        return {"answer": "Error!", "source": None, "namespace": pinecone_namespace, "id": "", "prompt_token_size": "0", "success": False, "error_message": error_message}          

init()

if __name__ == "__main__":

    #full_url = "https://www.deghi.it/supporto/"
    #full_url= "https://gethelp.tiledesk.com/"
    #full_url = "https://docs.pinecone.io/"

    #full_url = "https://knowledge.webafrica.co.za"
    #full_url = "https://aulab.it/"
    #full_url = "https://ecobaby.it/"
    #full_url = "https://lineaamica.gov.it/" # TRY IT LONGER
    #full_url = "http://cairorcsmedia.it/"
    #full_url = "https://www.sace.it/"
    #full_url = "https://www.ucl.ac.uk/"
    #full_url = "http://iostudiocongeco.it/"
    #full_url = "https://www.postpickr.com/"
    #full_url = "https://www.loloballerina.com/"
    full_url = "temp-namespace"

    if full_url == "temp-namespace":
        question_list = [
                        "what time is it?",
                        "who is Mustafa Kemal ?",
                        "what are the pricing for tiledesk?",
        ]

        for question in question_list:
            answer = main(question=question, openai_api_key=variables_db.OPENAI_API_KEY, namespace=full_url, model="gpt-3.5-turbo")
        
    if full_url == "https://gethelp.tiledesk.com/":
        question_list = [
        "How can I connect Tiledesk to WhatsApp?",
        "What is Tiledesk?",
        "How to integrate Tiledesk with my CRM?",
        "Which JavaScript code should I use for installing the Tiledesk widget on my website?",
        "How to automate responses in Tiledesk?",
        "Can I use Tiledesk for managing multiple teams?",
        "How to connect Tiledesk with Telegram?",
        "What are the pricing plans for Tiledesk?",
        "Is there a mobile app available for Tiledesk?",
        "How do I customize the Tiledesk chat interface?",
        "Can Tiledesk handle chatbots for customer service?",
        "How to set up multilingual support in Tiledesk?",
        "What analytics features does Tiledesk offer?",
        "How can I import existing customer support tickets into Tiledesk?",
        "What kind of APIs does Tiledesk offer for integration?",
        "How secure is customer data in Tiledesk?",
        "Can Tiledesk be used for video calls?",
        "How does Tiledesk handle offline messages?",
        "What are the options for customer feedback collection in Tiledesk?",
        "How to add custom fields to Tiledesk contact forms?",
        "How can I export reports from Tiledesk?",
        "Which ports need to be open for Tiledesk's operation?",
        "How to retrieve the full chat history via Tiledesk API?",
        "What webhook events are available in Tiledesk?",
        "How to handle file uploads through the Tiledesk widget?",
        "What are the limitations of the free version of Tiledesk?",
        "Good morning, what are the support hours for Tiledesk?",
        "Hello, could you guide me through the setup process?",
        "What day is it?",
        "Ciao",
        "How to manage agent permissions in Tiledesk?",
        "Can Tiledesk integrate with social media platforms?",
        "What are the best practices for setting up an effective knowledge base in Tiledesk?",
        "What's the weather like today?",
        "Can you recommend a good book to read?",
        "What's the best way to learn a new language?",
        "How do I bake a chocolate cake?",
        "Are there any good movies playing this weekend?",
        "What's the latest score of the football game?",
        "How do I start learning yoga?",
        "What are the seven wonders of the world?",
        "Can you give me some tips for photography?",
        "How do I reset my phone when it freezes?",
        "What are some healthy dinner options?",
        "Can you explain the theory of relativity?",
        "What are the trending topics on social media today?"
    ]
        question_list_old = [
            "How can I export reports from Tiledesk?",
            "How to add custom fields to Tiledesk contact forms?"

    ]
        for question in question_list:
            answer = main(question=question, openai_api_key=variables_db.OPENAI_API_KEY, namespace=full_url, model="gpt-3.5-turbo")


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
            answer = main(question=question, openai_api_key=variables_db.OPENAI_API_KEY, namespace=full_url, model="gpt-3.5-turbo")



    elif full_url == "https://docs.pinecone.io/":
        question = "What is pinecone?"
        answer = main(question=question, openai_api_key=variables_db.OPENAI_API_KEY, namespace=full_url, model="gpt-3.5-turbo")

        print(f"Question: {question}\nAnswer: {answer}")

        question = "How to create an index?"
        answer = main(question=question, openai_api_key=variables_db.OPENAI_API_KEY, namespace=full_url, model="gpt-3.5-turbo")

        print(f"Question: {question}\nAnswer: {answer}")
    elif full_url == "https://knowledge.webafrica.co.za":
        question_list = [
                        "Why should i use webafrica?",
                        
        ]
        for question in question_list:
            answer = main(question=question, openai_api_key=variables_db.OPENAI_API_KEY, namespace=full_url, model="gpt-3.5-turbo")
        
    elif full_url == "https://ecobaby.it/":
        question_list = [
                        "What is ecobaby?",
                        
        ]
        for question in question_list:
            answer = main(question=question, openai_api_key=variables_db.OPENAI_API_KEY, namespace=full_url, model="gpt-3.5-turbo")

    elif full_url == "https://aulab.it/":
        question_list = [
                        "What is web programming?",
                      
        ] 
        for question in question_list:
            answer = main(question=question, openai_api_key=variables_db.OPENAI_API_KEY, namespace=full_url, model="gpt-3.5-turbo")

    elif full_url == "https://lineaamica.gov.it/":
        question_list = [
                        "What is web lineaamicia?",
                      
        ] 
        for question in question_list:
            answer = main(question=question, openai_api_key=variables_db.OPENAI_API_KEY, namespace=full_url, model="gpt-3.5-turbo")


    elif full_url == "http://cairorcsmedia.it/":
        question_list = [
                        "What is cairorcs media?",
                      
        ] 
        for question in question_list:
            answer = main(question=question, openai_api_key=variables_db.OPENAI_API_KEY, namespace=full_url, model="gpt-3.5-turbo")


    elif full_url == "https://www.sace.it/":
        question_list = [
                        "What is sace?",
                      
        ] 
        for question in question_list:
            answer = main(question=question, openai_api_key=variables_db.OPENAI_API_KEY, namespace=full_url, model="gpt-3.5-turbo")


    elif full_url == "https://www.ucl.ac.uk/":
        question_list = [
                        "What is ucl?",
                      
        ] 
        for question in question_list:
            answer = main(question=question, openai_api_key=variables_db.OPENAI_API_KEY, namespace=full_url, model="gpt-3.5-turbo")


    elif full_url == "http://iostudiocongeco.it/":
        question_list = [
                        "What is iostudiocongeco?",
                      
        ] 
        for question in question_list:
            answer = main(question=question, openai_api_key=variables_db.OPENAI_API_KEY, namespace=full_url, model="gpt-3.5-turbo")


    elif full_url == "https://www.postpickr.com/":
        question_list = [
                        "What is postpickr?",
                      
        ] 
        for question in question_list:
            answer = main(question=question, openai_api_key=variables_db.OPENAI_API_KEY, namespace=full_url, model="gpt-3.5-turbo")


    elif full_url == "https://www.loloballerina.com/":
        question_list = [
                        "What is loloballerina?",
                      
        ] 
        for question in question_list:
            answer = main(question=question, openai_api_key=variables_db.OPENAI_API_KEY, namespace=full_url, model="gpt-3.5-turbo")


