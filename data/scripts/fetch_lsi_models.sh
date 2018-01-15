#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" && pwd )"
echo $DIR
cd $DIR

FILE=lsi_models.tar.gz
URL=https://www.dropbox.com/s/mh9ozxkforhi3rd/lsi_models.tar.gz?dl=0
CHECKSUM=545b954874f3a12c75f3515a91bc9698

if [ -f $FILE ]; then
  echo "File already exists. Checking md5..."
  os=`uname -s`
  if [ "$os" = "Linux" ]; then
    checksum=`md5sum $FILE | awk '{ print $1 }'`
  elif [ "$os" = "Darwin" ]; then
    checksum=`cat $FILE | md5`
  fi
  if [ "$checksum" = "$CHECKSUM" ]; then
    echo "Checksum is correct. No need to download."
    exit 0
  else
    echo "Checksum is incorrect. Need to download again."
  fi
fi

echo "Downloading LSI Faster R-CNN demo model (509M)..."

wget $URL -O $FILE

echo "Unzipping..."

mkdir -p "lsi_models"
tar -C "lsi_models" -zxvf $FILE

echo "Done. Please run this command again to verify that checksum = $CHECKSUM."
