FROM python:3.7

COPY ./api /api/api
COPY requirements.txt .

RUN pip install -r requirements.txt

ENV PYTHONPATH=/api

WORKDIR /api

EXPOSE 8000

ENTRYPOINT ["uvicorn"]
CMD ["api.main:app" , "--host", "0.0.0.0"]