#!/bin/bash 
DATA_DIR=$(dirname $1)
SHUFFLED_FILE=$DATA_DIR/shuffled_$(basename $1)
echo "shuffled file = $SHUFFLED_FILE"
head -1 $1 > $SHUFFLED_FILE
sed 1d $1 | shuf >> $SHUFFLED_FILE
echo "completed"