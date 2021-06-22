#!/bin/sh
wget -P /opt/ml/input/data/ https://prod-aistages-public.s3.ap-northeast-2.amazonaws.com/app/Competitions/000043/data/train_dataset.zip
cd /opt/ml/input/data
unzip train_dataset.zip
cd ~
# wget -P /opt/ml/input/data/sample/ https://drive.google.com/u/0/uc?id=1EjSfLv-eb-nWAW-kZuJOyWQFOy0uY1aE&export=download

# curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1EjSfLv-eb-nWAW-kZuJOyWQFOy0uY1aE" > /dev/null
# curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1EjSfLv-eb-nWAW-kZuJOyWQFOy0uY1aE" -o /opt/ml/input/data/new_data