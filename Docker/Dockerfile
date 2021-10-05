FROM rdeng2614/cerebro:initial_image
WORKDIR /root
RUN git clone https://github.com/mc2-project/cerebro.git
WORKDIR /root/cerebro
RUN git submodule update --init --recursive
WORKDIR /root/cerebro/crypto_backend/emp-toolkit/emp-tool
RUN git pull origin master && cmake . && make && sudo make install
WORKDIR /root/cerebro/crypto_backend/emp-toolkit/emp-ot
RUN cmake . && make && sudo make install
WORKDIR /root/cerebro/crypto_backend/emp-toolkit/emp-agmpc
RUN git pull origin master
RUN cmake . && make
WORKDIR /root
RUN apt-get update && apt-get install -y \
    python-gmpy2 \
    python-pip \
    emacs
COPY requirements.txt /root
RUN pip install -r requirements.txt