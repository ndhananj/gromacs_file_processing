################################################################################
# module to process files of various forms
# original author: Nithin Dhananjayan (ndhananj@ucdavis.edu)
################################################################################

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import re

import os

from biopandas.pdb import PandasPdb


################################################################################
# General functions
################################################################################

# Return the first two group matches for the first search string that matches
#     It is assumed each search string produces two group matches
def return_2_match_groups(search_strings,string_to_search):
    f=lambda x:re.search(x,string_to_search)
    ms=[f(x) for x in search_strings]
    fulls = [ (m.group(1), m.group(2)) for m in ms if m ]
    return fulls[:][0][:]

################################################################################
# xvg related functions
################################################################################
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
    # read and split data into parts
    f = open(filename, 'r+')
    lines = f.read().split("\n")
    num_lines = len(lines)
    comments = [lines[i] for i in range(num_lines-1) if lines[i][0]=='#']
    params = [lines[i] for i in range(num_lines-1) if lines[i][0]=='@']
    data = [lines[i].split() \
        for i in range(num_lines-1) if lines[i][0] not in ['@','#']]
    data_array = np.array(data).astype(np.float)
    # process and interpret the parts
    num_y_data_cols = data_array.shape[1]-1
    processed_params = process_xvg_params(params)
    y_labels = processed_params['yaxis  label']
    # use labels to place the data into a data frame
    num_y_labels = len(y_labels)
    num_new_labels = num_y_data_cols-num_y_labels
    if(num_new_labels>0): # if there aren't enough labels
        print("Extending labels to "+str(num_y_data_cols)+" ...")
        new_labels = [y_labels[-1]+"_"+str(i) for i in range(num_new_labels)]
        y_labels.extend(new_labels)
        print(y_labels)
    num_y_labels = len(y_labels)
    # create dictionary to make data frame
    # start with y labels
    data_pairs = [(y_labels[i], data_array[:,i+1]) for i in range(num_y_labels)]
    # add the x label
    x_label = processed_params['xaxis  label']
    data_pairs.insert(0,(x_label,data_array[:,0]))
    # turn intoa dictionary
    data_dict = {k:v for (k,v) in data_pairs}
    # turn into a data frame
    df = pd.DataFrame(data=data_dict)
    return {'data':df, 'xaxis label':x_label, 'yaxis labels':y_labels}

########
# assuming an xvg with an x_label and the rest as num_y_labels
# The input is the return value from reading the xvg

# returns  the average y_label values
def xvg_ylabel_avg(xvg_data):
    return xvg_data['data'][xvg_data['yaxis labels']].mean(axis=0)

# returns the first y_label values
def xvg_ylabel_first(xvg_data):
    return xvg_data['data'][xvg_data['yaxis labels']].iloc[0]

# returns the avg shift from first y_label values
def xvg_ylabel_shift(xvg_data):
    return xvg_ylabel_avg(xvg_data) - xvg_ylabel_first(xvg_data)

################################################################################
# Trajectory conversion functions
################################################################################

def convert_gro_to_pdb(file_prefix):
    gro_file = file_prefix+'.gro'
    pdb_file = file_prefix+'.pdb'
    cmd = "echo '0' | gmx trjconv -f "+gro_file+' -s '+gro_file+' -o '+pdb_file
    os.system(cmd)

def get_frames_from_trj(trj_file, struct_file, beg, end, step, out_prefix):
    cmd = "echo '0' | gmx trjconv -f "+trj_file+" -s "+struct_file+" -b "+\
        str(beg)+ " -e "+str(end)+" -dt "+\
        str(step)+" -sep -o "+out_prefix+".pdb"
    os.system(cmd)

def extract_position_from_traj_using_index(trj_file,struct_file,ndx_file,ndx,\
    output_file):
    cmd = "echo "+str(ndx)+" | gmx trajectory -f "+trj_file+" -s "+struct_file+\
        ' -n '+ndx_file+' -ox '+output_file
    os.system(cmd)

################################################################################
# Simulation functions
################################################################################

#These functions assume the forfield is set up to be available in directories

def grompp(mdp_file,init_struct_file,top_file,ndx_file,output_file):
    cmd = 'gmx grompp -f '+mdp_file+' -c '+init_struct_file+ \
        ' -p '+top_file+' -n '+ndx_file+' -o '+output_file
    os.system(cmd)

def mdrun(tpr_file,prefix):
    cmd = 'gmx mdrun -s '+tpr_file+' -deffnm '+prefix
    os.system(cmd)
