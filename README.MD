# chat playrgound.AI

Portale per analizzare i dataset in maniera semplice

## Requirements

- Docker
- Docker Compose

## Getting Started

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







