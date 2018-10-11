#!/bin/bash

rsync -ruv --exclude data --exclude features mel23:/mnt/FS1/caleb/PPGDataAnalysis/local-cache/* ./local-cache/
