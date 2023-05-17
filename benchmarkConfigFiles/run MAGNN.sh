#!/bin/bash

dataset="RegPattern"
#dataset="Yelp"
#dataset="RPvec"
folder="data/${dataset}/"
node_file="${folder}node.dat"
link_file="${folder}link.dat"
path_file="${folder}path.dat"
label_file="${folder}label.dat"
emb_file="${folder}emb.dat"

size=256
adim=100
nhead=8
nlayer=2
rtype="RotatE0"
dropout=0.5

nepoch=100
batchsize=512
sampling=100
lr=0.005
weight_decay=0.001

device="cpu"
attributed="False"
supervised="False"

python3 src/main.py --node=${node_file} --link=${link_file} --path=${path_file} --label=${label_file} --output=${emb_file} --device=${device} --hdim=${size} --adim=${adim} --nhead=${nhead} --nlayer=${nlayer} --rtype=${rtype} --dropout=${dropout} --nepoch=${nepoch} --batchsize=${batchsize} --sampling=${sampling} --lr=${lr} --weight-decay=${weight_decay} --attributed=${attributed} --supervised=${supervised}
#python3 src/main.py --node=${node_file} --link=${link_file} --output=${emb_file} --device=${device} --hdim=${size} --adim=${adim} --nhead=${nhead} --nlayer=${nlayer} --rtype=${rtype} --dropout=${dropout} --nepoch=${nepoch} --batchsize=${batchsize} --sampling=${sampling} --lr=${lr} --weight-decay=${weight_decay} --attributed=${attributed} --supervised=${supervised}
