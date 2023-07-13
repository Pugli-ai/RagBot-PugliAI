import pandas as pd
import openai
# import api_key
import pandas as pd
# import api_key
import numpy as np
from dotenv import load_dotenv
import os 

from openai.embeddings_utils import distances_from_embeddings

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

def create_context(
    question, df, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        
        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4
        
        # If the context is too long, break
        if cur_len > max_len:
            break
        
        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)

def answer_question(
    df,
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
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )

    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the question and context
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
        
        # Get the source URL for the most similar context
        source_url = df.loc[df['distances'].idxmin(), 'url']
        

        # Add the source URL to the answer
        answer = response["choices"][0]["text"].strip()
        dunno_list = ["I don't know", "I don’t know", "I do not know", "I don’t know", "I don't know.", "I don’t know.", "I do not know.", "I don’t know."]
        if answer in dunno_list:
            source_url = None

        
        return {"answer": answer, "source_url": source_url}
    except Exception as e:
        print(e)
        return {"answer": "Error!", "source_url": None}
    

if __name__=="__main__":

    df=pd.read_csv('processed/embeddings.csv', index_col=0)
    df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

    question = "What day is it?"
    answer = answer_question(df, question=question, debug=False)
    print(f"Question: {question}\nAnswer: {answer}")

    question="which javascript code should i copy and paste for installing the widget on my website ? please write me that javascript code"
    answer = answer_question(df, question=question, debug=False)
    print(f"Question: {question}\nAnswer: {answer}")

    question="How to connect Tiledesk with Telegram"
    answer = answer_question(df, question=question, debug=False)
    print(f"Question: {question}\nAnswer: {answer}")