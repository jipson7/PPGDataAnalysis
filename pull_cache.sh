#!/bin/bash

rsync -ruv --exclude features mel23:/mnt/FS1/caleb/ppg/local-cache/* ./local-cache/
