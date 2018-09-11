#!/bin/bash

rsync -ruv --exclude 'training_data/temp'  mel:/scratch/caleb/ppg/data-cache/* ./data-cache/
