#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 10:59:28 2022

@author: Daniel
"""

import pandapower as pp
import pandas as pd
import numpy as np
import tempfile
import os
from pandapower.timeseries import DFData
from pandapower.timeseries import OutputWriter
from pandapower.timeseries.run_time_series import run_timeseries
from pandapower.control import ConstControl
import matplotlib.pyplot as plt

# Build power network
def build_net():
    """
    Build the base configuration of the power system under study
    Input: None
    Output: pandapower network
    """    
    net = pp.create_empty_network()

    # Buses
    bus1 = pp.create_bus(net, name="CLARK-Region1", vn_kv=110, type="b")
    bus2 = pp.create_bus(net, name="AMHERST-Region1", vn_kv=110, type="b")
    bus3 = pp.create_bus(net, name="WINLOCK-Region1", vn_kv=110, type="b")
    bus4 = pp.create_bus(net, name="BOWMAN-Region2", vn_kv=110, type="b")
    bus5 = pp.create_bus(net, name="TROY-Region2", vn_kv=110, type="b")
    bus6 = pp.create_bus(net, name="MAPLE-Region2", vn_kv=110, type="b")
    bus7 = pp.create_bus(net, name="GRAND-Region3", vn_kv=110, type="b")
    bus8 = pp.create_bus(net, name="WAUTAGA-Region3", vn_kv=110, type="b")
    bus9 = pp.create_bus(net, name="CROSS-Region3", vn_kv=110, type="b")

    # Line
    line1 = pp.create_line(net, bus1, bus4, length_km=10, std_type="149-AL1/24-ST1A 110.0", name="Line 1-4")
    line2 = pp.create_line(net, bus4, bus9, length_km=10, std_type="149-AL1/24-ST1A 110.0", name="Line 4-9")
    line3 = pp.create_line(net, bus8, bus9, length_km=10, std_type="149-AL1/24-ST1A 110.0", name="Line 8-9")
    line4 = pp.create_line(net, bus2, bus8, length_km=10, std_type="149-AL1/24-ST1A 110.0", name="Line 2-8")
    line5 = pp.create_line(net, bus7, bus8, length_km=10, std_type="149-AL1/24-ST1A 110.0", name="Line 7-8")
    line6 = pp.create_line(net, bus6, bus7, length_km=10, std_type="149-AL1/24-ST1A 110.0", name="Line 6-7")
    line7 = pp.create_line(net, bus3, bus6, length_km=10, std_type="149-AL1/24-ST1A 110.0", name="Line 3-6")
    line8 = pp.create_line(net, bus5, bus6, length_km=10, std_type="149-AL1/24-ST1A 110.0", name="Line 5-6")
    line9 = pp.create_line(net, bus4, bus5, length_km=10, std_type="149-AL1/24-ST1A 110.0", name="Line 4-5")

    # Gen
    gen1_nom_MW = 0
    gen2_nom_MW = 163
    gen3_nom_MW = 85

    gen1 = pp.create_gen(net, bus1, p_mw=gen1_nom_MW, slack=True, name="Generator 1") #PU bus or Slack bus
    gen2 = pp.create_sgen(net, bus2, p_mw=gen2_nom_MW, q_mvar=0, name="Generator 2")
    gen3 = pp.create_sgen(net, bus2, p_mw=gen3_nom_MW, q_mvar=0, name="Generator 3")

    # Load
    load1_nom_MW = 90
    load1_nom_MVar = 30
    load2_nom_MW = 100
    load2_nom_MVar = 35
    load3_nom_MW = 125
    load3_nom_MVar = 50

    load1 = pp.create_load(net, bus5, p_mw=load1_nom_MW, q_mvar=load1_nom_MVar, name="Load 1")
    load2 = pp.create_load(net, bus7, p_mw=load2_nom_MW, q_mvar=load2_nom_MVar, name="Load 2")
    load4 = pp.create_load(net, bus9, p_mw=load3_nom_MW, q_mvar=load3_nom_MVar, name="Load 3")

    return net


# Timeseries Power Flow Implementation
# Randomize load profiles
np.random.seed(0) #Uncomment to get different randomized load profiles
def create_data_source(net, op_state='', n_timesteps=60):
    """
    Creating data sources of different load conditions (high, low, normal) by varying P and Q of load
    Input: none
    Output: dataframe of datasource
    """
    profiles = pd.DataFrame()
    if op_state == 'Normal Load':
        for i in range(len(net.load)):
            profiles['load{}_P'.format(str(i))] = net.load.p_mw[i] + (0.05 * np.random.random(n_timesteps) * net.load.p_mw[i])
            profiles['load{}_Q'.format(str(i))] = net.load.q_mvar[i] + (0.05 * np.random.random(n_timesteps) * net.load.q_mvar[i])
    elif op_state == 'High Load':
        for i in range(len(net.load)):
            profiles['load{}_P'.format(str(i))] = 1.2 * net.load.p_mw[i] + (0.05 * np.random.random(n_timesteps) * net.load.p_mw[i])
            profiles['load{}_Q'.format(str(i))] = 1.2 * net.load.q_mvar[i] + (0.05 * np.random.random(n_timesteps) * net.load.q_mvar[i])
    elif op_state == 'Low Load':
        for i in range(len(net.load)):
            profiles['load{}_P'.format(str(i))] = 0.75 * net.load.p_mw[i] + (0.05 * np.random.random(n_timesteps) * net.load.p_mw[i])
            profiles['load{}_Q'.format(str(i))] = 0.75 * net.load.q_mvar[i] + (0.05 * np.random.random(n_timesteps) * net.load.q_mvar[i])

    ds = DFData(profiles)

    return profiles, ds


def create_controllers(net, ds):
    """
    Declaring constant controller for changing the P and Q values of sgen/PQ bus and load
    Input: pandapower network, datasource
    Output: updated pandapower network
    """
    for i in range(len(net.load)):
        ConstControl(net, element='load', variable='p_mw', element_index=[i],
                     data_source=ds, profile_name=['load{}_P'.format(str(i))])
        ConstControl(net, element='load', variable='q_mvar', element_index=[i],
                     data_source=ds, profile_name=['load{}_Q'.format(str(i))])
    
    return net


def create_output_writer(net, time_steps, output_dir):
    """
    Declaring OutputWriter to save results of timeseries
    Input: pandapower network, number of timesteps, directory where excel file will be saved
    Output: output writer of timeseries analysis
    """
    ow = OutputWriter(net, time_steps, output_path=output_dir, output_file_type=".xls", log_variables=list())
    # these variables are saved to the harddisk after / during the time series loop
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_bus', 'va_degree')
    return ow


# run_timeseries per operation state
net = build_net()
n_time_steps = 60


# Normal Load
def normal_load(net, n_time_steps, output_dir):
    _net = net
    profiles, ds = create_data_source(_net, op_state='Normal Load', n_timesteps=n_time_steps)
    _net = create_controllers(_net, ds)
    time_steps = range(0, n_time_steps)
    ow = create_output_writer(_net, time_steps, output_dir)
    run_timeseries(_net, time_steps, calculate_voltage_angles=True)

# High Load
def high_load(net, n_time_steps, output_dir):
    _net = net
    profiles, ds = create_data_source(_net, op_state='High Load', n_timesteps=n_time_steps)
    _net = create_controllers(_net, ds)
    time_steps = range(0, n_time_steps)
    ow = create_output_writer(_net, time_steps, output_dir)
    run_timeseries(_net, time_steps, calculate_voltage_angles=True)

# Low Load
def low_load(net, n_time_steps, output_dir):
    _net = net
    profiles, ds = create_data_source(_net, op_state='Low Load', n_timesteps=n_time_steps)
    _net = create_controllers(_net, ds)
    time_steps = range(0, n_time_steps)
    ow = create_output_writer(_net, time_steps, output_dir)
    run_timeseries(_net, time_steps, calculate_voltage_angles=True)


# Gen 3 disconnected during High Load
def high_load_gen3_discon(net, n_time_steps, output_dir):
    _net = net
    index_sgen = pp.get_element_index(_net, 'sgen', 'Generator 3')    
    _net.sgen.in_service[index_sgen] = False
    profiles, ds = create_data_source(_net, op_state='High Load', n_timesteps=n_time_steps)
    _net = create_controllers(_net, ds)
    time_steps = range(0, n_time_steps)
    ow = create_output_writer(_net, time_steps, output_dir)
    run_timeseries(_net, time_steps, calculate_voltage_angles=True)
    _net.sgen.in_service[index_sgen] = True

# Gen 3 disconnected during Low Load
def low_load_gen3_discon(net, n_time_steps, output_dir):
    _net = net
    index_sgen = pp.get_element_index(_net, 'sgen', 'Generator 3')
    _net.sgen.in_service[index_sgen] = False
    profiles, ds = create_data_source(_net, op_state='Low Load', n_timesteps=n_time_steps)
    _net = create_controllers(_net, ds)
    time_steps = range(0, n_time_steps)
    ow = create_output_writer(_net, time_steps, output_dir)
    run_timeseries(_net, time_steps, calculate_voltage_angles=True)
    _net.sgen.in_service[index_sgen] = True

# Line 8 (bus5-bus6) disconnected during High Load
def high_load_line8_discon(net, n_time_steps, output_dir):
    _net = net
    index_line = pp.get_element_index(_net, 'line', 'Line 5-6')
    _net.line.in_service[index_line] = False
    profiles, ds = create_data_source(_net, op_state='High Load', n_timesteps=n_time_steps)
    _net = create_controllers(_net, ds)
    time_steps = range(0, n_time_steps)
    ow = create_output_writer(_net, time_steps, output_dir)
    run_timeseries(_net, time_steps, calculate_voltage_angles=True)
    _net.line.in_service[index_line] = True

# Line 8 (bus5-bus6) disconnected during Low Load
def low_load_line8_discon(net, n_time_steps, output_dir):
    _net = net
    index_line = pp.get_element_index(_net, 'line', 'Line 5-6')
    _net.line.in_service[index_line] = False
    profiles, ds = create_data_source(_net, op_state='Low Load', n_timesteps=n_time_steps)
    _net = create_controllers(_net, ds)
    time_steps = range(0, n_time_steps)
    ow = create_output_writer(_net, time_steps, output_dir)
    run_timeseries(_net, time_steps, calculate_voltage_angles=True)
    _net.line.in_service[index_line] = True



output_dir_normal_load = os.path.join(tempfile.gettempdir(), "time_series", "normal_load")
print("Results can be found in your local temp folder: {}".format(output_dir_normal_load))
if not os.path.exists(output_dir_normal_load):
    os.mkdir(output_dir_normal_load)
normal_load(net, n_time_steps, output_dir_normal_load)
net = build_net()

output_dir_high_load = os.path.join(tempfile.gettempdir(), "time_series", "high_load")
print("Results can be found in your local temp folder: {}".format(output_dir_high_load))
if not os.path.exists(output_dir_high_load):
    os.mkdir(output_dir_high_load)
high_load(net, n_time_steps, output_dir_high_load)
net = build_net()

output_dir_low_load = os.path.join(tempfile.gettempdir(), "time_series", "low_load")
print("Results can be found in your local temp folder: {}".format(output_dir_low_load))
if not os.path.exists(output_dir_low_load):
    os.mkdir(output_dir_low_load)
low_load(net, n_time_steps, output_dir_low_load)
net = build_net()

output_dir_high_load_gen3_dis = os.path.join(tempfile.gettempdir(), "time_series", "high_load_gen3_dis")
print("Results can be found in your local temp folder: {}".format(output_dir_high_load_gen3_dis))
if not os.path.exists(output_dir_high_load_gen3_dis):
    os.mkdir(output_dir_high_load_gen3_dis)
high_load_gen3_discon(net, n_time_steps, output_dir_high_load_gen3_dis)
net = build_net()

output_dir_low_load_gen3_dis = os.path.join(tempfile.gettempdir(), "time_series", "low_load_gen3_dis")
print("Results can be found in your local temp folder: {}".format(output_dir_low_load_gen3_dis))
if not os.path.exists(output_dir_low_load_gen3_dis):
    os.mkdir(output_dir_low_load_gen3_dis)
low_load_gen3_discon(net, n_time_steps, output_dir_low_load_gen3_dis)
net = build_net()

output_dir_high_load_line8_dis = os.path.join(tempfile.gettempdir(), "time_series", "high_load_line8_dis")
print("Results can be found in your local temp folder: {}".format(output_dir_high_load_line8_dis))
if not os.path.exists(output_dir_high_load_line8_dis):
    os.mkdir(output_dir_high_load_line8_dis)
high_load_line8_discon(net, n_time_steps, output_dir_high_load_line8_dis)
net = build_net()

output_dir_low_load_line8_dis = os.path.join(tempfile.gettempdir(), "time_series", "low_load_line8_dis")
print("Results can be found in your local temp folder: {}".format(output_dir_low_load_line8_dis))
if not os.path.exists(output_dir_low_load_line8_dis):
    os.mkdir(output_dir_low_load_line8_dis)
low_load_line8_discon(net, n_time_steps, output_dir_low_load_line8_dis)


# Now we want to read the excel files for voltage pu and angle and merge
# The data in a panda file format
# Normal Load
file_vm_pu_normal_load = os.path.join(output_dir_normal_load, "res_bus", "vm_pu.xls")
vm_pu_normal_load = pd.read_excel(file_vm_pu_normal_load, index_col=0)

file_va_degree_normal_load = os.path.join(output_dir_normal_load, "res_bus", "va_degree.xls")
va_degree_normal_load = pd.read_excel(file_va_degree_normal_load, index_col=0)

normal_load_df = pd.concat([vm_pu_normal_load, va_degree_normal_load], axis=1, ignore_index=True)
normal_load_df['state'] = 'normal load'

# High Load
file_vm_pu_high_load = os.path.join(output_dir_high_load, "res_bus", "vm_pu.xls")
vm_pu_high_load = pd.read_excel(file_vm_pu_high_load, index_col=0)

file_va_degree_high_load = os.path.join(output_dir_high_load, "res_bus", "va_degree.xls")
va_degree_high_load = pd.read_excel(file_va_degree_high_load, index_col=0)

high_load_df = pd.concat([vm_pu_high_load, va_degree_high_load], axis=1, ignore_index=True)
high_load_df['state'] = 'high load'

# Low Load
file_vm_pu_low_load = os.path.join(output_dir_low_load, "res_bus", "vm_pu.xls")
vm_pu_low_load = pd.read_excel(file_vm_pu_low_load, index_col=0)

file_va_degree_low_load = os.path.join(output_dir_low_load, "res_bus", "va_degree.xls")
va_degree_low_load = pd.read_excel(file_va_degree_low_load, index_col=0)

low_load_df = pd.concat([vm_pu_low_load, va_degree_low_load], axis=1, ignore_index=True)
low_load_df['state'] = 'low load'

# Gen3 disconnected during High Load
file_vm_pu_high_load_gen3_dis = os.path.join(output_dir_high_load_gen3_dis, "res_bus", "vm_pu.xls")
vm_pu_high_load_gen3_dis = pd.read_excel(file_vm_pu_high_load_gen3_dis, index_col=0)

file_va_degree_high_load_gen3_dis = os.path.join(output_dir_high_load_gen3_dis, "res_bus", "va_degree.xls")
va_degree_high_load_gen3_dis = pd.read_excel(file_va_degree_high_load_gen3_dis, index_col=0)

high_load_gen3_dis_df = pd.concat([vm_pu_high_load_gen3_dis, va_degree_high_load_gen3_dis], axis=1, ignore_index=True)
high_load_gen3_dis_df['state'] = 'high load gen3 dis'

# Gen3 disconnected during Low Load
file_vm_pu_low_load_gen3_dis = os.path.join(output_dir_low_load_gen3_dis, "res_bus", "vm_pu.xls")
vm_pu_low_load_gen3_dis = pd.read_excel(file_vm_pu_low_load_gen3_dis, index_col=0)

file_va_degree_low_load_gen3_dis = os.path.join(output_dir_low_load_gen3_dis, "res_bus", "va_degree.xls")
va_degree_low_load_gen3_dis = pd.read_excel(file_va_degree_low_load_gen3_dis, index_col=0)

low_load_gen3_dis_df = pd.concat([vm_pu_low_load_gen3_dis, va_degree_low_load_gen3_dis], axis=1, ignore_index=True)
low_load_gen3_dis_df['state'] = 'low load gen3 dis'

# Line8 (bus5-bus6) disconnected during High Load
file_vm_pu_high_load_line8_dis = os.path.join(output_dir_high_load_line8_dis, "res_bus", "vm_pu.xls")
vm_pu_high_load_line8_dis = pd.read_excel(file_vm_pu_high_load_line8_dis, index_col=0)

file_va_degree_high_load_line8_dis = os.path.join(output_dir_high_load_line8_dis, "res_bus", "va_degree.xls")
va_degree_high_load_line8_dis = pd.read_excel(file_va_degree_high_load_line8_dis, index_col=0)

high_load_line8_dis_df = pd.concat([vm_pu_high_load_line8_dis, va_degree_high_load_line8_dis], axis=1, ignore_index=True)
high_load_line8_dis_df['state'] = 'high load line8 dis'

# Line8 (bus5-bus6) disconnected during Low Load
file_vm_pu_low_load_line8_dis = os.path.join(output_dir_low_load_line8_dis, "res_bus", "vm_pu.xls")
vm_pu_low_load_line8_dis = pd.read_excel(file_vm_pu_low_load_line8_dis, index_col=0)

file_va_degree_low_load_line8_dis = os.path.join(output_dir_low_load_line8_dis, "res_bus", "va_degree.xls")
va_degree_low_load_line8_dis = pd.read_excel(file_va_degree_low_load_line8_dis, index_col=0)

low_load_line8_dis_df = pd.concat([vm_pu_low_load_line8_dis, va_degree_low_load_line8_dis], axis=1, ignore_index=True)
low_load_line8_dis_df['state'] = 'low load line8 dis'



dataset = pd.concat([normal_load_df, high_load_df, low_load_df, high_load_gen3_dis_df, low_load_gen3_dis_df
                    , high_load_line8_dis_df, low_load_line8_dis_df], 
                    axis=0, ignore_index=True)

print(np.shape(dataset))

def plot_simulation_result():
    fig, ax = plt.subplots(nrows=7, figsize=(7, 12))
    # Plotting
    color = ['tab:red', 'tab:orange', 'tab:cyan', 'tab:green', 'tab:blue',
                 'tab:brown', 'tab:pink', 'tab:gray', 'tab:purple']
    voltage_df = [vm_pu_normal_load , vm_pu_high_load, vm_pu_low_load,
                      vm_pu_high_load_gen3_dis, vm_pu_low_load_gen3_dis,
                      vm_pu_high_load_line8_dis, vm_pu_low_load_line8_dis]
    angle_df = [va_degree_normal_load, va_degree_high_load, va_degree_low_load,
                    va_degree_high_load_gen3_dis, va_degree_low_load_gen3_dis,
                    va_degree_high_load_line8_dis, va_degree_low_load_line8_dis]
    title_list = ['Base topopology, normal load',
                    'Base topoplogy, high load',
                    'Base topology, low load',
                    'Gen 3 disconnected, high load',
                    'Gen 3 disconnected, low load',
                    'Line 5-6 disconnected, high load',
                    'Line 5-6 disconnected, low load']
    for j in range(0, 7):
            for i in range(0, 9):
                ax[j].scatter(voltage_df[j][i], angle_df[j][i], c=color[i], s=5, label='Bus {}'.format(i + 1))
                box = ax[j].get_position()
                ax[j].set_position([box.x0, box.y0, box.width, box.height])
                ax[j].set_title(title_list[j])
                ax[j].set_xlabel('Voltage')
                ax[j].set_ylabel('Angle')

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right',ncol=1, fancybox=True, shadow=True)
    plt.show()
    fig_file_name = 'plot.png'
    fig.savefig(fig_file_name)

plot = plot_simulation_result()
dataset_for_normalization = dataset.drop(['state'], axis=1)
data_normalized = dataset.copy()
print(dataset)

for i in range(1, 9):
    data_normalized[i] = np.divide(dataset_for_normalization[i] - dataset_for_normalization[i].min(),
                                      dataset_for_normalization[i].max() - dataset_for_normalization[i].min())

for i in range(10, 18):
    data_normalized[i] = np.divide(dataset_for_normalization[i] - dataset_for_normalization[i].min(),
                                      dataset_for_normalization[i].max() - dataset_for_normalization[i].min())

dataset_norm_labeled = data_normalized.copy()
dataset_norm_labeled['state'] = dataset['state'].copy()

dataset.to_excel("dataset.xlsx")
dataset_norm_labeled = dataset_norm_labeled.sample(frac=1).reset_index(drop=True)
dataset_norm_labeled.to_excel("dataset_norm_labeled.xlsx")



