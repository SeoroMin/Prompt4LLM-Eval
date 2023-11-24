docker run -it -h name \
        -p external_port:in_port \
        --ipc=host \
        --name Container_name \
        --gpus all \
        -v path \
        anaconda bash
