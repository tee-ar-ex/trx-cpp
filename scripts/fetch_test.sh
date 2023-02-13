#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/..

function hashcheck() {
  MD5=$(md5sum "$1" | cut -d " " -f1)
  if [[ $MD5 != $2 ]]
  then
    echo "md5sum for $1 does not match. Please remove the file to download it again."
    exit 1
  fi
}

mkdir -p data

curl -L https://figshare.com/ndownloader/files/37624154 --output data/DSI.zip
hashcheck data/DSI.zip b847f053fc694d55d935c0be0e5268f7 # V1 (27.09.2022)

curl -L https://figshare.com/ndownloader/files/37624148 --output data/memmap_test_data.zip
hashcheck data/memmap_test_data.zip 03f7651a0f9e3eeabee9aed0ad5f69e1 # V2 (27.09.2022)

curl -L https://figshare.com/ndownloader/files/37624151 --output data/trx_from_scratch.zip
hashcheck data/trx_from_scratch.zip d9f220a095ce7f027772fcd9451a2ee5 # V2 (27.09.2022)
