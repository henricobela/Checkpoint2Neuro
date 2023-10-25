FROM python:3.10

COPY . .

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt
RUN chmod +x docker_entrypoint.sh

ENTRYPOINT ["/bin/bash" , "./docker_entrypoint.sh"]