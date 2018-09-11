#!/bin/bash

 rsync -ruv --exclude 'data-cache' ./* swift155:~/ppg/
