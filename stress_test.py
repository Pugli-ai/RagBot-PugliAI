import asyncio
import aiohttp
import json

# The URL for the API endpoint
url = "https://tiledesk-prod-v2.bravesky-de61454e.francecentral.azurecontainerapps.io/api/qa"

# The data you want to send in the POST request
data = {
    "question": "what do you know about mosul issue?",
    "namespace": "kerem-name",
    "gptkey": "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",
    "model": "gpt-3.5-turbo"
}

# The headers for the POST request
headers = {
    "Content-Type": "application/json",
    # "Authorization": "Bearer YOUR_ACCESS_TOKEN",  # Uncomment and replace if needed
}

async def make_request(session, url, data):
    async with session.post(url, data=json.dumps(data), headers=headers) as response:
        return await response.json()

async def make_requests(url, data, amount):
    async with aiohttp.ClientSession() as session:
        tasks = [make_request(session, url, data) for _ in range(amount)]
        return await asyncio.gather(*tasks)

# Run the async function to make 100 requests
responses = asyncio.run(make_requests(url, data, 100))

# Now responses is a list of all the responses
for response in responses:
    print(response)
