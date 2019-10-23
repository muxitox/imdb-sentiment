#!/bin/bash

file_name=`python -c "import datetime; print(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))"`
mkdir -p ./out/$1/$file_name
sbatch --output=./out/$1/$file_name/log.out --error=./out/$1/$file_name/log.err ./launchers/$1.sh "$1/$file_name"
