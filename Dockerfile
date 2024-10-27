FROM pytorch/pytorch:latest

# prevent apt from hanging
ARG DEBIAN_FRONTEND=noninteractive

ENV HOME=/workspace
WORKDIR $HOME

# general system dependencies:
RUN apt update
RUN apt install -y git
RUN apt install libgl1-mesa-glx -y
RUN apt install libglib2.0-0 -y
RUN apt install wget -y

ENV REPODIR=$HOME/deep_pavements_lite

# this repository:
COPY . $REPODIR
WORKDIR $REPODIR

# for mapillary downloading (submodule):
RUN git clone https://github.com/kauevestena/my_mappilary_api.git
RUN git submodule init
RUN git submodule update
RUN pip install -r my_mappilary_api/requirements.txt

# CLIP:
RUN pip install ftfy regex tqdm
RUN pip install git+https://github.com/openai/CLIP.git

# other requirements:
RUN pip install -r requirements.txt

# getting deep pavements model:
RUN wget https://huggingface.co/kauevestena/clip-vit-base-patch32-finetuned-surface-materials/resolve/main/model.pt?download=true -O deep_pavements_clip_model.pt

# precaching (default, but it can be skipped):
ENV TO_PRECACHE=true

RUN if [ "$TO_PRECACHE" = "true" ]; then \
    python precaching/precaching_clip.py && \
    python precaching/precaching_oneformer.py; \
    fi

