FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# prevent apt from hanging
ARG DEBIAN_FRONTEND=noninteractive

ENV HOME=/workspace
WORKDIR $HOME

# general system dependencies:
RUN apt-get update && \
    apt-get install -y git libgl1 libgl1-mesa-dri libglib2.0-0 wget && \
    rm -rf /var/lib/apt/lists/*

ENV REPODIR=$HOME/deep_pavements_lite

# this repository:
COPY . $REPODIR
WORKDIR $REPODIR

# for mapillary downloading (submodule):
RUN git submodule update --init --recursive
RUN pip install --no-cache-dir -r my_mappilary_api/requirements.txt

# all requirements (includes CLIP, ML deps, geospatial, etc.):
RUN pip install --no-cache-dir -r requirements.txt

# getting deep pavements model (optional):
RUN wget https://huggingface.co/kauevestena/clip-vit-base-patch32-finetuned-surface-materials/resolve/main/model.pt?download=true -O deep_pavements_clip_model.pt || echo "Warning: Could not download model, will use default CLIP model"

# precaching (default, but it can be skipped):
ENV TO_PRECACHE=true

RUN if [ "$TO_PRECACHE" = "true" ]; then \
    (python modules/precaching/precaching_clip.py || echo "Warning: CLIP precaching failed") && \
    (python modules/precaching/precaching_oneformer.py || echo "Warning: OneFormer precaching failed"); \
    fi
