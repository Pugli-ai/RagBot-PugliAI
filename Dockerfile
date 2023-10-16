FROM python:3.8-slim-buster
COPY ./backend/requirements.txt .
RUN pip install -r requirements.txt

# Install Firefox
RUN apt-get update && apt-get install -y firefox-esr

# Copy geckodriver to the container
COPY ./backend/geckodriver /geckodriver

WORKDIR /api

COPY ./backend/api /api/api

ENV PYTHONPATH=/api
EXPOSE 8000

ENTRYPOINT ["uvicorn"]
CMD ["api.main:app" , "--host", "0.0.0.0"]