import asyncio
import aiohttp
import json

# Constants
SCRAPER_SINGLE_API_URL = "https://tiledesk-dev.blackwave-d2bf4ee1.westus2.azurecontainerapps.io/api/scrape/single"
QA_API_URL = "https://tiledesk-dev.blackwave-d2bf4ee1.westus2.azurecontainerapps.io/api/qa"
GPT_KEY = "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"
NAMESPACE = "kerem-namespace2"
SCRAPING_URLS = [
    "https://tiledesk.com/",
    "https://tiledesk.com/qualified-leads-generation/",
    "https://tiledesk.com/chatbot-for-customer-service/",
    "https://tiledesk.com/chatbot-template/",
    "https://tiledesk.com/pricing-live-chat/",
    "https://tiledesk.com/blog/",
    "https://tiledesk.com/whatsapp-chatbot-integration/",
    "https://tiledesk.com/facebook-integration-with-chatbot/",
    "https://tiledesk.com/free-live-chat-widget-for-wordpress-website/",
    "https://tiledesk.com/best-chatbot-for-your-shopify-store/",
]
QA_QUESTION = "what is Design Studio?"

async def scrape_single(session, url, index):
    data = {
        "id": f"temp-id{index}",  # Unique ID for each page
        "content": "",
        "source": url,
        "type": "url",
        "gptkey": GPT_KEY,
        "namespace": NAMESPACE
    }
    async with session.post(SCRAPER_SINGLE_API_URL, json=data) as response:
        return await response.json()

async def qa_request(session):
    data = {
        "question": QA_QUESTION,
        "namespace": NAMESPACE,
        "gptkey": GPT_KEY
    }
    async with session.post(QA_API_URL, json=data) as response:
        return await response.json()

async def main():
    async with aiohttp.ClientSession() as session:
        scrape_tasks = [scrape_single(session, url, index) for index, url in enumerate(SCRAPING_URLS)]
        qa_tasks = [qa_request(session) for _ in range(20)]
        results = await asyncio.gather(*scrape_tasks, *qa_tasks)
        for result in results:
            print(json.dumps(result, indent=2))

# Run the async main function
asyncio.run(main())
