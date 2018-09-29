#!/bin/bash

rsync -ruv --exclude 'local-cache' --exclude '__pycache__' ./* mel:/scratch/caleb/ppg
