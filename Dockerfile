FROM python:3.9.7-slim

RUN apt update -y && apt install awscli -y
RUN pip install -U pip
RUN pip install pipenv 


WORKDIR /app

RUN apt-get update && apt-get install -y build-essential wget
#gcc wget g++ make cmake autoconf ninja-build
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
  tar -xvzf ta-lib-0.4.0-src.tar.gz && \
  cd ta-lib/ && \
  ./configure --prefix=/usr --build=aarch64-unknown-linux-gnu && \
  make && \
  make install



COPY . /app
# COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy
RUN rm -R ta-lib ta-lib-0.4.0-src.tar.gz

# CMD ["python3","app.py"]
COPY [ "src/stockSelProj/pipeline/predict.py", "./" ]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "app:app" ]