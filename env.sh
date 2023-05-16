#!/bin/bash

install_requirements() {
    file="$1"

    while read -r line; do
        conda install -y $line

        # $? == return val of "conda install", successfully installed if $? == 0
        if [ $? -ne 0 ]; then
            # use pip if conda doesnt work
            pip install $line
        fi
    done < "$file"
}

install_requirements requirements.txt