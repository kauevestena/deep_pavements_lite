FROM pytorch/pytorch:latest

# prevent apt from hanging
ARG DEBIAN_FRONTEND=noninteractive

ENV HOME /workspace

WORKDIR $HOME
ENV REPODIR $HOME/deep_pavements_lite

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



