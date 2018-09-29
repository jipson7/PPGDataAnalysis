#!/bin/bash

rsync -ruv --exclude features mel22:/scratch/caleb/ppg/local-cache/* ./local-cache/
