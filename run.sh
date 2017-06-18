#!/bin/sh

cd ./build
./saliency --input "../data/input/" "../data/saliency/"
cd $OLDPWD

cd ./tools
./eval.sh
cd $OLDPWD 


