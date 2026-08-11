#!/bin/sh
set -e
clang -O2 -o branch branch.c
clang -O2 -o cache cache.c
python3 analyze.py
