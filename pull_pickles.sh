#!/bin/bash

rsync -ruv --exclude 'training_data/temp'  swift155:~/ppg/data-cache/* ./data-cache/
