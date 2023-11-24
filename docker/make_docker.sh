docker run -it -h sangmin \
        -p external_port:in_port \
        --ipc=host \
        --name Container_name \
        --gpus all \
        -v path \
        anaconda bash
