# deep_pavements_lite

This version of Deep Pavements is an all-in-one package ("batteries-included"), making use of a fixed-class network for the semantic segmentation part.

# Docker Setup

1) Clone and setup:

    git clone https://github.com/kauevestena/deep_pavements_lite
    cd deep_pavements_lite
    git submodule update --init --recursive

2) Build the Docker image:

    docker build --tag 'deep_pavements_lite' .

To skip the precaching of weights, add `--build-arg TO_PRECACHE=false`

3) Run the Docker container:

    MOUNT_FOLDER="$HOME/data/deep_pavements_lite"
    mkdir -p $MOUNT_FOLDER
    echo "<YOUR MAPILLARY TOKEN>" > $MOUNT_FOLDER/mapillary_token

    docker run --name running_d_p_l \
        -v $MOUNT_FOLDER:/workspace/data \
        --gpus all \
        -it 'deep_pavements_lite' \
        python runner.py \
            --lat_min <min_latitude> \
            --lon_min <min_longitude> \
            --lat_max <max_latitude> \
            --lon_max <max_longitude>

(Or If you wanna use it inside VSCode, as a dev container, you can use the "devcontainer.json" at the .devcontainer folder, but don't forget to create the folder and the token file!)

Inside the container you can create a file called "workspace/data/mapillary_token" and put your token there. Or simply on repo root, both are accepted.