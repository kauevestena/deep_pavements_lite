# deep_pavements_lite

This version of Deep Pavements is all-in-one package ("batteries-included"), making use of a fixed-class network for the semantic segmentation part.


# Docker Setup

first, buid:

    docker build --tag 'deep_pavements_lite' .

then run (please note that the mounted host folder is a suggestion, you can change it, but "workspace/data" must be used):

    docker run --name running_d_p_l -v $HOME/data/deep_pavements_lite:/workspace/data --gpus all -it 'deep_pavements_lite' 

(Or If you wanna use it inside VSCode, as a dev container, you can use the "devcontainer.json" at the .devcontainer folder)