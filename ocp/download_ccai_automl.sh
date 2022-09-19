#!/bin/bash

# Training set 
python scripts/download_data.py --task s2ef --split 200k --get-edges --num-workers 8 --ref-energy
# Validation sets
python scripts/download_data.py --task s2ef --split val_id --get-edges --num-workers 8 --ref-energy
	# TODO

# Test set (LARGE)


