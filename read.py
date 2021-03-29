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
    count = 0
    for doc in all_data:
        if "id" not in doc:
            doc["id"] = "id-"+str(count)
            count += 1
    return all_data
            

