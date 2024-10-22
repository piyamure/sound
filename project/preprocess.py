"""
1. read segy file: this is file format seismic data was saved
2. save data into numpy array: if needed 
3. get sound amplitude data from the source (for a given shot)
4. get sound amplitude data at each receiver line: (for a given shot)
4. get time stamps (assuming equal time stamps across data)

"""
import numpy as np
import pandas as pd
import scipy
import os
import segyio

# read segy file 
def read_segy(sgyfile,textheader=None):
    dataout = None
    sampint = None
    textheader_output = None
    
    with segyio.open(sgyfile, "r", ignore_geometry=True) as f:
        # Get the number of traces and sample size
        num_traces = f.tracecount     # Number of traces 
        num_samples = f.samples.size  # Number of samples per trace
        print(f'No of traces = {num_traces}, No of samples per trace = {num_samples}')
        
        # Initialize dataout array
        dataout = np.zeros((num_traces, num_samples))
        
        # Populate the dataout with traces
        for i in range(num_traces):
            dataout[i, :] = f.trace[i]  # Fill each row with the trace data
        
        # Extracting sample interval
        delta_t = f.samples[1] - f.samples[0]  # Assuming regular sampling
        
        # Extracting text header if requested
        if textheader == 'yes':
            textheader_output = segyio.tools.wrap(f.text[0])
            
    return dataout, delta_t, textheader_output

# save data to numpy file (if nedded)
def save_data_to_numpy(data, data_path):
    np.save(data_path,data)
    
# source indices from data
def get_source_indices(num_shots,num_lines,receivers_per_line):
    source_idx = np.array([(receivers_per_line *num_lines)*i + i for i in range(num_shots +1)])
    source_idx = np.concatenate((np.array([-1]),source_idx))
    return source_idx[1:]

# get source signal 
def get_source_signal(data,shot_no,soure_indices):
    return data[soure_indices[shot_no]]

# signal from receiver line 
def get_receiver_line_data(data,shot_no,line_no,soure_indices,num_lines):
    signal = data[soure_indices[shot_no]+1:soure_indices[shot_no +1]] 
    line_signals = np.split(signal,num_lines,axis=0)           
    return np.array(line_signals[line_no])

# get time stamps: data=[idx,time]
def get_time(data,delta_t=2*1e-3):
    return np.arange(data.shape[1])*delta_t

# normalize signal
def min_max_normalize(array,min_val=-1,max_val=1):
    norm_array = (array - array.min())/(array.max() - array.min())
    norm_array = norm_array * (max_val - min_val) + min_val
    return norm_array

'''
# de-normalize signal and save min/max
def denormalize(norm_array,original_min,original_max,min_val=-1,max_val=1):
    array = (norm_array - min_val)/(max_val - min_val)
    array = array*(original_max - original_min) + original_min
    return array '''


    
if __name__ == '__main__':

    SGY_PATH = 'data/R1809SA8S299.sgy'
    NUMPY_SAVE_PATH = 'data/data.npy'
    
    NUM_SHOTS = 3
    RECEIVERS_PER_LINE= 638 
    NUM_LINES = 8 
    
    data,delta_t,_ = read_segy(SGY_PATH)
    source_idxs = get_source_indices(NUM_SHOTS,NUM_LINES,RECEIVERS_PER_LINE)
    print(f'source_idx = {source_idxs}')
    
    source1_signal = get_source_signal(data,shot_no=0,soure_indices=source_idxs)
    print(f'source_1_signal_shape = {source1_signal.shape}')
    
    receiver_line_8 = get_receiver_line_data(data,shot_no=0,line_no=7,soure_indices=source_idxs,num_lines=NUM_LINES)
    print(f'receiver_line_10_shape = {receiver_line_8.shape}')
