# deep_pavements_lite

This version of Deep Pavements is all-in-one package ("batteries-included"), making use of a fixed-class network for the semantic segmentation part.


# Docker Setup

first, buid (remember to change the token):

    docker build --tag 'deep_pavements_lite' .

To skip the precaching of weights, add `--build-arg TO_PRECACHE=false`

then run (please note that the mounted host folder is a suggestion, you can change it, but "workspace/data" must be used!):

    MOUNT_FOLDER="$HOME/data/deep_pavements_lite"

    mkdir -p $MOUNT_FOLDER

    # don't forget to set the token!
    echo "<YOUR MAPILLARY TOKEN>" > $MOUNT_FOLDER/mapillary_token

    docker run --name running_d_p_l -v $MOUNT_FOLDER:/workspace/data --gpus all -it 'deep_pavements_lite' 

(Or If you wanna use it inside VSCode, as a dev container, you can use the "devcontainer.json" at the .devcontainer folder, but don't forget to create the token file!)