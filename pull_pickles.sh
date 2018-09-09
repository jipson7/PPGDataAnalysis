#!/bin/bash

rsync -ruv --exclude 'training_data/temp'  logan:~/ppg/data-cache/* ./data-cache/
