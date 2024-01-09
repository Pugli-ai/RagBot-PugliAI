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
# **Definition of REST Body and Response Parameters**

<<<<<<< HEAD
## List of REST APIs
- Scrapers 
  - Scrape Single
  - Scrape Tree
- Q&A
- Scraper Status
- List Namespace
- Delete
  - Delete by ID
  - Delete by Namespace

# Scrapers
## Scrape Single

=======
# **Definition of REST Body and Response Parameters**

## List of REST APIs
- Scrapers 
  - Scrape Single
  - Scrape Tree
- Q&A
- Scraper Status
- List Namespace
- Delete
  - Delete by ID
  - Delete by Namespace

# Scrapers
## Scrape Single

>>>>>>> e5fa4ee (change background task to queue on single scrape)
**API Endpoint**: `https://tiledesk-dev.blackwave-d2bf4ee1.westus2.azurecontainerapps.io/api/scrape/single`

#### Request Body:
```json
{
    "id": "temp-id57",
    "content": "",
    "source": "https://tiledesk.com/pricing-live-chat/",
    "type": "url",
    "gptkey": "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",
    "namespace": "kerem-namespace"
}
```
#### Fields Description:
- `id`: The identifier of the single resource.
- `content`: The content data to save in Pinecone as a row. Should be an empty string (`""`) if `type` is url
- `source`: The source of the content. Should be a URL if the `type` key is "url". If not, it can be any string, such as "file.txt".
- `type`: Either "url" or "text". Set as "url" if `content` is not provided and `source` is a URL. Set as "text" if `content` is provided and `source` is not a URL.
- `gptkey`: The API key for OpenAI services.
- `namespace`: The namespace of the knowledgebase in Pinecone. This is used for grouping datasets within the knowledgebase.

### Response:
```json
{
    "message": "Scrape request added to queue! Check scraping status API for progress."
}
``` 

#### Fields Description:
- `message`: Provides detailed information about the scraping request that has been ordered. This message indicates that the request has been queued and advises users to check the scraping status API for updates on the progress.

## Scrape Tree

**API Endpoint**: `https://tiledesk-dev.blackwave-d2bf4ee1.westus2.azurecontainerapps.io/api/scrape/tree`

### Request Body:
```json
{
    "full_url" : "https://tiledesk.com/",
    "gptkey" : "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",
    "namespace": "tiledesk-tree"
}
```

#### Fields Description:
- `full_url`: URL of the root page that should be scraped along with its subpages.
- `gptkey`: API key for OpenAI services.
- `namespace`: The namespace in Pinecone where the knowledgebase data will be saved. This is used for grouping datasets within the knowledgebase.

 
### Response:
```json
{
    "message": "Scrape request added to queue! Check scraping status API for progress."
}
``` 

#### Fields Description:
- `message`: A notification indicating that the scrape request has been added to the queue. Users are advised to check the scraping status API for updates on the progress.

# Q&A
**API Endpoint**: `https://tiledesk-dev.blackwave-d2bf4ee1.westus2.azurecontainerapps.io/api/qa`

### Request Body:
```json
{
    "question": "what is Design Studio?",
    "namespace": "tiledesk-tree",
<<<<<<< HEAD
    "gptkey": "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",
    "model": "gpt-3.5-turbo",
=======
    "gptkey": "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"
>>>>>>> e5fa4ee (change background task to queue on single scrape)
}

``` 

_Fields description_:
- _question_: question asked by the user
- _namespace_: knowledge base namespace on pinecone
- _gptkey_: openai gpt key
<<<<<<< HEAD
- _model_: gpt model name . "gpt-3.5-turbo" or "gpt-4"
=======
>>>>>>> e5fa4ee (change background task to queue on single scrape)

### Response:
```json
{
    "answer": "Design Studio is an advanced drag-and-drop platform that brings your dream conversation to life. It allows you to build your ideal bot from scratch or use ready-made templates designed specifically for automating lead generation and customer support processes.",
    "source": "https://tiledesk.com/nl/no-code-chatbot-builder/",
    "namespace": "tiledesk-tree",
    "id": "https://tiledesk.com/nl/no-code-chatbot-builder/#0",
    "success": true,
    "error_message": null
}
```
#### Fields Description:
- `answer`: Answer returned by the action based on the question and the context.
- `source`: The source of the content.
- `namespace`: The namespace of the source (knowledgebase group ID).
- `id`: The ID of the source on Pinecone.
- `success`: A boolean that is true if the action can generate an answer, or false if it cannot.
- `error_message`: Message returned if an error occurs.


# Scraping Status

**API Endpoint**: `https://tiledesk-dev.blackwave-d2bf4ee1.westus2.azurecontainerapps.io/api/scrape/status`

### Request Body:
```json
{
    "namespace_list": [],
    "namespace": "kerem-namespace",
    "id": "temp-id57"
}

//////////  OR  //////////
 
{
    "namespace_list": [
        "flowiseai",
        "langchain",
        "pinecone",
        "deghi"
    ],
    "namespace": "",
    "id": ""
}
```

#### Fields Description:
- `namespace_list`: A list of namespaces for the knowledgebases to check. You can provide a single namespace, but it must be within a list. This field should be an empty list if checking the status of a single scrape.
- `namespace`: The namespace of the knowledgebase for a single scrape. This field should be an empty string if checking the status of a scrape tree.
- `id`: The ID of the knowledge for a single scrape. This field should be an empty string if checking the status of a scrape tree.

### Response for scrape single status:
```json
{
    "status_message": "Crawling is finished for temp-id57",
    "status_code": 3,
    "queue_order": -1
}
```

#### Fields Description:
- `status_message`: The message indicating the scraping status.
- `status_code`: The status code of scraping, where the values represent the following:
    - 0: Not started
    - 2: Started
    - 3: Finished
    - 4: Error
- `queue_order`: Since the single scrape status does not involve a queue, this field will always be -1. This is maintained for structural consistency.

### Response for Namespace List:
```json
{
    "flowiseai": {
        "status_message": "Crawling is queued in the 1st order for the flowiseai.",
        "status_code": 1,
        "queue_order": 1
    },
    "langchain": {
        "status_message": "Crawling is queued in the 2nd order for the langchain.",
        "status_code": 1,
        "queue_order": 2
    },
    "pinecone": {
        "status_message": "Crawling is started but not finished yet for pinecone",
        "status_code": 2,
        "queue_order": 0
    },
    "deghi": {
        "status_message": "Crawling is finished for deghi",
        "status_code": 3,
        "queue_order": -1
    }
}
``` 

#### Fields Description:
- `status_message`: The message indicating the current scraping status.
- `status_code`: The status code of scraping, defined as follows:
    - 0: Not started
    - 1: Queued
    - 2: Started
    - 3: Finished
    - 4: Error
- `queue_order`: The position of the website in the queue. A value of 0 means it is currently being processed, while -1 indicates that it is not in the queue.


# List Namespace

**API Endpoint**: `https://tiledesk-dev.blackwave-d2bf4ee1.westus2.azurecontainerapps.io/api/list/namespace`

### Request Body:
```json
{
    "namespace": "kerem-namespace"
}
```

#### Fields Description:
- `namespace`: The namespace of the knowledgebase that will be used for retrieving all data.

### Response:
```json
{
    "success": true,
    "message": {
        "temp-id1": {
            "content": "source: https://www.google.com/ Chi siamo Google Store GmailImmagini Accedi Google offerto in: English Italia Pubblicità Soluzioni aziendali Come funziona la Ricerca Il nostro terzo decennio di azione per il clima Privacy Termini Impostazioni IT Accedi Prima di continuare su Google Ricordiamo che i nostri..",
            "id": "temp-id1",
            "is_tree": "False",
            "n_tokens": 499.0,
            "source": "https://www.google.com/",
            "type": "url"
        },
        "temp-id0": {
            "content": "source: https://tiledesk.com/pricing-live-chat/ Discover Tiledesk’s Web to ChatGPT transformer! Solutions Products Pricing Resources Log in Sign up free Engage your customers now Please Select Your Plan Monthly Yearly..",
            "id": "temp-id0",
            "is_tree": "False",
            "n_tokens": 588.0,
            "source": "https://tiledesk.com/pricing-live-chat/",
            "type": "url"
        }
    }
}
``` 
#### Fields Description:
- `success`: A boolean value that indicates whether the given index was successfully deleted.
- `message`: The detailed information about the rows within the specified namespace.

Note: `is_tree` indicates whether the row was scraped using the scraper tree method or not.




# Delete ID

**API Endpoint**: `https://tiledesk-dev.blackwave-d2bf4ee1.westus2.azurecontainerapps.io/api/delete/id`

### Request Body:
```json
{
    "id": "temp-id56",
    "namespace": "kerem-namespace"  
}
```
#### Fields Description:
- `id`: The ID of the content on Pinecone.
- `namespace`: The namespace of the content to delete.

### Response:
``` json
{
    "success": true,
    "message": "temp-id56 is deleted from database"
}
``` 
#### Fields Description:
- `success`: A boolean value that indicates whether the given index was successfully deleted or not.
- `message`: The message detailing the outcome of the deleting process.


# Delete Namespace

**API Endpoint**: `https://tiledesk-dev.blackwave-d2bf4ee1.westus2.azurecontainerapps.io/api/delete/namespace`

### Request Body:
```json
{
    "namespace": "kerem-namespace"
}

```
#### Fields Description:
- `namespace`: The namespace of the knowledgebase to delete.

### Response:
```json
{
    "success": true,
    "message": "kerem-namespace is deleted from database"
}
``` 
#### Fields Description:
- `success`: A boolean value that indicates whether the given namespace was successfully deleted or not.
- `message`: The message detailing the outcome of the deleting process.

