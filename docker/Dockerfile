# syntax=docker.io/docker/dockerfile:1.7-labs

#
# docker build -f ./docker/Dockerfile -t atlasai/geochat:latest .
#
# docker tag \
# 	atlasai/diffsat-single-image:latest \
# 	us-west1-docker.pkg.dev/data-processor-prod-375818/geochat/geochat:latest
# docker push \
# 	us-west1-docker.pkg.dev/data-processor-prod-375818/geochat/geochat:latest
#
# docker run -it --rm -p 8080:8080 --gpus all --entrypoint /bin/bash atlasai/geochat:latest
#

FROM nvidia/cuda:12.3.2-base-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG QT_QPA_PLATFORM=offscreen

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
ENV CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

ENV PIP_NO_CACHE_DIR=1

ENV APPLICATION_VENV="geochat"
ENV SOURCE_ROOT="/GeoChat"
ENV WEIGHTS_ROOT="/geochat-7B"
ENV OUTPUT_DIR="/output"

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git wget curl bzip2 unzip gnupg2 sudo && \
    apt-get install -y lsb-release software-properties-common

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs && \
    git lfs install
 
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /tmp

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -f && \
    rm -rf Miniconda3-latest-Linux-x86_64.sh && \
    . ~/miniconda3/bin/activate && \
    conda init --all

RUN . ~/miniconda3/bin/activate && \
    conda create -y -n ${APPLICATION_VENV} python=3.10 && \
    echo "source activate ${APPLICATION_VENV}" >> ~/.bashrc && \
    conda clean -a -y && \
    conda activate ${APPLICATION_VENV} && \
    pip install --upgrade pip && \
    pip install ipython opencv-python-headless

# we are copying local checkout of weights
#git clone https://huggingface.co/MBZUAI/geochat-7B && \
WORKDIR "${WEIGHTS_ROOT}"
COPY ./weights/ "${WEIGHTS_ROOT}"

WORKDIR /

ADD https://api.github.com/repos/feiws/balance/git/refs/heads/main /tmp/version.json
RUN rm -rf /tmp/version.json && \
    git clone https://github.com/AtlasAIPBC/GeoChat.git && \
    . ~/miniconda3/bin/activate && \
    cd GeoChat && \
    conda activate ${APPLICATION_VENV} && \
    pip install -e .

WORKDIR "${SOURCE_ROOT}"
ENTRYPOINT ["/bin/bash"]
