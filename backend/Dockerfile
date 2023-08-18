FROM python:3.8-slim-buster
COPY ./requirements.txt .
RUN pip install -r requirements.txt
WORKDIR /api
COPY ./api /api/api
ENV PYTHONPATH=/api
EXPOSE 8000
ENTRYPOINT ["uvicorn"]
CMD ["api.main:app" , "--host", "0.0.0.0"]