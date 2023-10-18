# Q&A Bot for websites with generative ai 

## Requirements

- Docker
- Docker Compose
- check the requirements.txt file for the python packages

## Getting Started Locally

1. **Clone the repository**:
```bash
git clone https://github.com/Pugli-ai/chat-playground.git
cd chat-playground
```

2. **Build and run the Docker container**:
```bash
docker compose -f docker-compose.yml up --build
# docker compose -f docker-compose.yml up backend --build

To run it without docker:
# cd backend
# uvicorn api.main:app --host 0.0.0.0
```

3. **Connect to the application**:
Access the app locally at http://localhost:8100.

# Getting started with Cloud API

#**Definition of REST body and response parameters**
List of REST API:
- scarping
- q&a
- scraper status
- delete namespace

# Scarping
API: https://tiledesk-app.gentleisland-369885c7.westus2.azurecontainerapps.io/api/scrape

### _Body_:
``` 
{
    "full_url" : "https://www.deghi.it/supporto/",
    "gptkey" : ""
    
}
```
_Fields description_:
- full_url: url to scrape
- gptkey: openai gpt key
 
### _Response_:
``` 
{
    "message": "Scrape request added to queue! Check scraping status API for progress."
}
``` 
_Fields description_:
- message: information about scraping added to the queue

# Q&A
API: https://tiledesk-app.gentleisland-369885c7.westus2.azurecontainerapps.io/api/qa
### _Body_:
``` 
{
    "question": "How about whatsapp ? ",
    "kbid" : "https://gethelp.tiledesk.com/",
    "gptkey" : "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",
    "chat_history_dict" : {
        "0": {
            "question": "How to integrate to my website? ",
            "answer": "To integrate Tiledesk to your website, follow these steps: \n1. Locate your unique Tiledesk javascript code on your Tiledesk Dashboard by going to Settings > Widget > Add chat to your website..."
        },
        "1": {
            "question": "How about telegram?",
            "answer": "To connect Tiledesk with Telegram, follow these steps:\n\n1. Login in Tiledesk Dashboard, then click “Apps” in the left menu and click Install on the Telegram icon..."
        }
    }
}
``` 

_Fields description_:
- _question_: question asked by the user in tiledesk chatbot
- _kbid_: knowledge base id represents the link to the website to Q&A
- _gptkey_: openai gpt key
- _chat_history_dict_: previous answers and questions asked by the user with the action. if there is no chat history, give an empty dictionary "chat_history_dict" : {} .

### _Response_:
```
{
    "answer": "To move a conversation from a web widget to WhatsApp chat, you need to go to the Chatbot Design Studio in Tiledesk.",
    "source_url": "https://gethelp.tiledesk.com/tags/how-to-send-bulk-whatsapp-messages-for-free/",
    "success": true,
    "error_message": null
}
```
_Fields description_:
- _answer_: answer returned by the action based on question and the context
- _source_url_: link to the original source of the answer in the website
- _success_: boolean that is true if the action can generate an answer of false if it cannot
- _error_message_: message returned if an error occurs

## Chat History Dictionary structure
```
"chat_history_dict" : {
        "0": {
            "question": "How to integrate to my website? ",
            "answer": "To integrate Tiledesk to your website, follow these steps: \n1. Locate your unique Tiledesk javascript code on your Tiledesk Dashboard by going to Settings > Widget > Add chat to your website..."
        },
        "1": {
            "question": "How about telegram?",
            "answer": "To connect Tiledesk with Telegram, follow these steps:\n\n1. Login in Tiledesk Dashboard, then click “Apps” in the left menu and click Install on the Telegram icon..."
        }
    }
```


# Scarping status

API: https://tiledesk-app.gentleisland-369885c7.westus2.azurecontainerapps.io/api/scrape/status

### _Body_:
``` 
{
    "url_list" : [
        "https://docs.flowiseai.com/",
        "https://python.langchain.com/",
        "https://docs.pinecone.io/",
        "https://www.deghi.it/supporto/"
        ]
}
```
_Fields description_:
- _url_list_: url list of the websites to check (you can also give one url but it should be in the list) 
### _Response_:
``` 
{
    "https://docs.flowiseai.com/": {
        "status_message": "Crawling is queued in the 1. order for the https://docs.flowiseai.com/.",
        "status_code": 1,
        "queue_order": 1
    },
    "https://python.langchain.com/": {
        "status_message": "Crawling is queued in the 2. order for the https://python.langchain.com/.",
        "status_code": 1,
        "queue_order": 2
    },
    "https://docs.pinecone.io/": {
        "status_message": "Crawling is started but not finished yet for https://docs.pinecone.io/",
        "status_code": 2,
        "queue_order": 0
    },
    "https://www.deghi.it/supporto/": {
        "status_message": "Crawling is finished for https://www.deghi.it/supporto/",
        "status_code": 3,
        "queue_order": -1
    }
}
``` 
_Fields description_:
- _status_message_ : the message of the scraping status
- _status_code_: the status code of scraping, 0-> not started, 1-> queued, 2-> started, 3-> finished
- _queue_order_: the position of the website in queue (0 means it is processing, -1 means it is not in the queue)

# Delete namespace

API: https://tiledesk-app.gentleisland-369885c7.westus2.azurecontainerapps.io/api/scrape/delete
### _Body_:
``` 
{
    "full_url" : "https://www.deghi.it/supporto/"
}
```
_Fields description_:
- _full_url_: url of the website to delete 
### _Response_:
``` 
{
    "success": true,
    "message": "https://www.deghi.it/supporto/ is deleted from database"
}
``` 
_Fields description_:
- _success_: A bool value that indicates if given index deleted successfully or not
- _message_: the message of the deleting process

