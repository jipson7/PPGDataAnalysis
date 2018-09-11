#!/bin/bash

 rsync -ruv --exclude 'data-cache' ./* mel:/scratch/caleb/ppg
