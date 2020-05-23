################################################################################
# module to process files of various forms
# original author: Nithin Dhananjayan (ndhananj@ucdavis.edu)
################################################################################

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import re

from biopandas.pdb import PandasPdb


# Return the first two group matches for the first search string that matches
#     It is assumed each search string produces two group matches
def return_2_match_groups(search_strings,string_to_search):
    f=lambda x:re.search(x,string_to_search)
    ms=[f(x) for x in search_strings]
    fulls = [ (m.group(1), m.group(2)) for m in ms if m ]
    return fulls[:][0][:]

def process_xvg_params(params):
    search_strings = ['@(\w+)\s+(\w+)']
    search_strings.append('@\s+([\w\s]+)\s+\"(.+)\"')
    f=lambda x: return_2_match_groups(search_strings,x)
    param_pairs = [f(param) for param in params]
    y_axis_labels = [x[1] for x in param_pairs if x[0]=='yaxis  label']
    processed_params = {k:v for (k,v) in param_pairs }
    processed_params['yaxis  label'] = y_axis_labels
    return processed_params

def read_xvg(filename):
    f = open(filename, 'r+')
    lines = f.read().split("\n")
    num_lines = len(lines)
    comments = [lines[i] for i in range(num_lines-1) if lines[i][0]=='#']
    params = [lines[i] for i in range(num_lines-1) if lines[i][0]=='@']
    data = [lines[i].split() \
        for i in range(num_lines-1) if lines[i][0] not in ['@','#']]
    data_array = np.array(data).astype(np.float)
    processed_params = process_xvg_params(params)
    y_labels = processed_params['yaxis  label']
    num_y_labels = len(y_labels)
    data_pairs = [(y_labels[i], data_array[:,i+1]) for i in range(num_y_labels)]
    data_pairs.insert(0,(processed_params['xaxis  label'],data_array[:,0]))
    data_dict = {k:v for (k,v) in data_pairs}
    df = pd.DataFrame(data=data_dict)
    return df
