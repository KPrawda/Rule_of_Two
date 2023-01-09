# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 14:08:32 2023

@author: prawdak1
"""


import numpy as np  # Make sure NumPy is loaded before it is used in the callback
assert np  # avoid "imported but unused" message (W0611)
import scipy.io.wavfile
import scipy


def calculate_energy_median(signal):
    energy =  1.4826**2 * np.median(signal**2)
    
    return energy

def calculate_energy_mean(signal):
    energy = np.mean(np.abs(signal**2))
    
    return energy

def calculate_tv_factor(sw, noise, signal_energy):
    
    len_sweep, numSweeps = np.shape(sw)
    len_noise = np.shape(noise)[0]
    
    Ey1_y2 = np.zeros((numSweeps, numSweeps))
    En1_n2 = np.zeros((numSweeps, numSweeps))
    Ev1_v2 = np.zeros((numSweeps, numSweeps))
    tau = np.zeros((numSweeps, numSweeps))
    
    
    
    for i  in range(numSweeps):
        for j  in range(numSweeps):
            y1_y2 = sw[:, i] - sw[:, j]
            n1_n2 = noise[:, i] - noise[:, j]
            
            Ey1_y2[i,j] = calculate_energy_mean(y1_y2)/(len_sweep)
            En1_n2[i,j] = calculate_energy_median(n1_n2)/(len_noise)
            
            Ev1_v2[i,j] = Ey1_y2[i,j] - En1_n2[i,j]
            E_sv = Ev1_v2[i,j]/2
            E_sh[i] = signal_energy[ i] - E_sv
            
            tau[i,j] = np.abs(Ev1_v2[i,j]/E_sh[i])
            
    tv_factor = np.median(tau[tau>0])
    
    return tv_factor


def Ro2(PCC, threshold ):
    
    numSweeps = np.shape(PCC)[0]
    sweep_ind = np.arange(numSweeps)+1
    x = np.where(PCC < threshold)
    
    if np.size(x) == 0:
        print('All sweeps are clean (devoid of non-stationary noise)')
    else: 
        values, counts =np.unique(x, return_counts = True)
        ind = sweep_ind[counts > 3]
        
        print('Sweep(s) ',ind,' contain(s) non-stationary noise!')

# read the sweep file from the path
path = 'path_to_sweep_measurements'
filename_signal = 'sweep_file.wav'
fs, signal_ = scipy.io.wavfile.read(path+filename_signal)
sw = signal_#/np.max(np.absolute(signal_)) # normalize

filename_noise = 'noise_file.wav'
fs, noise = scipy.io.wavfile.read(path+filename_noise)


len_sweep, numSweeps = np.shape(sw)
len_noise = np.shape(noise)[0]

dims = np.shape(signal_)
numSweeps = dims[1] # number of sweeps in a channel


sw_energy= np.zeros([numSweeps])
noise_energy = np.zeros([numSweeps])
E_sh = np.zeros( numSweeps)

for i in range(numSweeps):    
    sw_energy[ i] = calculate_energy_mean(sw[:, i])/(len_sweep)
    noise_energy[i] = calculate_energy_median(noise[:, i])/(len_noise)
    
    
    

signal_energy = sw_energy-noise_energy    


# SNR-based thresholds
upper_bound = signal_energy/sw_energy
lower_bound = (signal_energy - noise_energy)/sw_energy

# transfer-function variation

tv_factor = calculate_tv_factor(sw, noise, signal_energy)

lower_bound_tv = lower_bound/(1+tv_factor)



PCC = np.corrcoef(sw,rowvar = False)
Ro2(PCC, lower_bound_tv)
