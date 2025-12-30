FROM python:3.12.12-bookworm

WORKDIR /app

# set port, our app working on 80
EXPOSE 80


# create environment
RUN python -m venv env
RUN ls
RUN . ./env/bin/activate 


RUN apt update && apt upgrade -y && apt install -y libgl1 libglib2.0-0

# setup reqs
COPY requirements.txt .
RUN pip install -U pip --root-user-action ignore
RUN pip install -r requirements.txt --root-user-action ignore


# copy project file
COPY ./app.py .
COPY ./OllamaModel.py .

COPY ./ollama_adapter/views/tools/bounding_box.py ./ollama_adapter/views/tools/bounding_box.py

#RUN echo "DATABASE_URL=postgresql://postgres:password@pg-obj-det:5432/postgres" > ./.env
#RUN echo "BUCKET_NAME=my-bucket" >> ./.env
#RUN echo "APP_AWS_ACCESS_KEY=random-key" >> ./.env
#RUN echo "APP_AWS_SECRET_KEY=random-key" >> ./.env
#RUN echo "APP_AWS_REGION=eu-central-1" >> ./.env
#RUN echo "DEV_AWS_ENDPOINT=http://localhost.localstack.cloud:4566" >> ./.env

RUN python -m chainlit create-secret | grep 'CHAINLIT_AUTH_SECRET' >> ./.env


# start app
CMD  python -m chainlit run --port 80 --host 0.0.0.0 -h app.py



