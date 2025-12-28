FROM python:3.12.12-slim

EXPOSE 8000

WORKDIR /modules/


COPY controllers/ ./ollama_adapter/controllers/.
COPY models/ ./ollama_adapter/models/.
COPY views/ ./ollama_adapter/views/.
COPY api.py ./ollama_adapter/.
COPY __init__.py ./ollama_adapter/.

COPY ../__init__.py ./.


ENV OLLAMA_HOST=http://host.docker.internal:11434

RUN pip install -U pip

COPY requirements.in .
RUN pip install -r requirements.in

#COPY requirements.txt .
#RUN pip install -r requirements.txt


CMD ["python3", "-m", "fastapi", "dev", "--host", "0.0.0.0", "--port", "80", "ollama_adapter/api.py"]

