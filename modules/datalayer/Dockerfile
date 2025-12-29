FROM debian:12.12-slim
WORKDIR /datalayer
COPY prisma .
RUN apt update 
RUN apt install -y nodejs npm 
RUN npm init -y 1>/dev/null 2>&1
#COPY ./env-postgres ./.env
