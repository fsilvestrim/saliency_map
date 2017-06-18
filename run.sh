#!/bin/sh

cd ./build
./saliency -i "../data/input" -o "../data/saliency/" -l 4 -c 3 -s 5
cd $OLDPWD

cd ./tools
./eval.sh
cd $OLDPWD 


