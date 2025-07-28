# имеющийся образ с ОС alpine
FROM python:3.12-alpine

# создается рабочий подкаталог 
WORKDIR /python-app

COPY requirements.txt .

RUN pip install requirements.txt

COPY . .

# команда запуска при развертывании контейнера
CMD [ "python", "main.py" ]