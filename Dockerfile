FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# prevent apt from hanging
ARG DEBIAN_FRONTEND=noninteractive

ENV HOME=/workspace
WORKDIR $HOME

# general system dependencies:
RUN apt-get update && \
    apt-get install -y git libgl1 libglib2.0-0 wget && \
    rm -rf /var/lib/apt/lists/*

ENV REPODIR=$HOME/deep_pavements_lite

# this repository:
COPY . $REPODIR
WORKDIR $REPODIR

# for mapillary downloading (submodule):
RUN git submodule update --init --recursive
RUN pip install -r my_mappilary_api/requirements.txt

# CLIP:
RUN pip install ftfy regex tqdm
RUN pip install git+https://github.com/openai/CLIP.git

# other requirements:
RUN pip install -r requirements.txt

# getting deep pavements model (optional):
RUN wget https://huggingface.co/kauevestena/clip-vit-base-patch32-finetuned-surface-materials/resolve/main/model.pt?download=true -O deep_pavements_clip_model.pt || echo "Warning: Could not download model, will use default CLIP model"

# precaching (default, but it can be skipped):
ENV TO_PRECACHE=true

RUN if [ "$TO_PRECACHE" = "true" ]; then \
    (python precaching/precaching_clip.py || echo "Warning: CLIP precaching failed") && \
    (python precaching/precaching_oneformer.py || echo "Warning: OneFormer precaching failed"); \
    fi
