FROM python:3.8-slim-buster
COPY ./requirements.txt .
COPY ./geckodriver .
RUN pip install -r requirements.txt
WORKDIR /api
RUN apt-get update
RUN apt-get install -y firefox-esr

COPY ./api /api/api
COPY /geckodriver .
ENV PYTHONPATH=/api
EXPOSE 8000
ENTRYPOINT ["uvicorn"]
CMD ["api.main:app" , "--host", "0.0.0.0"]