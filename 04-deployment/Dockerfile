FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

RUN pip install pipenv

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["starter.py", "./"]

ENTRYPOINT ["python", "starter.py"]