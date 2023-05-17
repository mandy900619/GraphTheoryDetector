#!/bin/bash

install_requirements() {
    file="$1"

    while read -r line || [[ -n $line ]]; do
        echo "INSTALL $line"

        conda install -y $line

        # $? == return val of "conda install", successfully installed if $? == 0
        if [ $? -ne 0 ]; then
            echo "c->p"
            # use pip if conda doesnt work
            pip install $line
        fi
    done < "$file"
}

# create conda env & install package
env_name="GraphTheoryDetector"
conda create -y -n $env_name
source ~/.bashrc
source activate
conda activate $env_name
install_requirements requirements.txt