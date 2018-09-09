#!/bin/bash

rsync -ruv --exclude 'data-cache' ./* logan:~/ppg/
