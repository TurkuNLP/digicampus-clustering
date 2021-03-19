#!/usr/bin/env python3

"""
Assumes the input to be json files
"""

import json

def read_files(filename_lst):
    all_data = []
    for fname in filename_lst:
        with open(fname, "r") as f:
            all_data.extend(json.load(f))
    return all_data
            

