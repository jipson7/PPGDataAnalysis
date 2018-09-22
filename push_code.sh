#!/bin/bash

rsync -ruv --exclude 'local-cache' ./* mel:/scratch/caleb/ppg
