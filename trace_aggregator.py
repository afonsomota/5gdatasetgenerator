import pandas as pd
import glob
import time
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from os import path
import os
import json
from shutil import copyfile
import random

from scipy.optimize import curve_fit
import pickle
import multiprocessing
from multiprocessing import Pool
        
    
import sys
import Util
from NRTables import BIDIR_MCS


def schedule_bet(df):
    in_df = df.copy()
    no_nodes = in_df.Node.count()
    scheduled_nodes = set()
    assert in_df["Bytes_scheduled_total"].nunique()==1
    bytes_to_schedule = in_df["Bytes_scheduled_total"].iloc[0]
    in_df["Bytes_scheduled"] = 0
    max_iter=no_nodes+500
    it = 0
    while True:
        it+=1
        max_iter -= 1
        if max_iter == 0:
            print("ERROR! Infinite loop. Bytes",bytes_to_schedule, in_df["Bytes_scheduled_total"].iloc[0], "nonodes",scheduled_nodes)
            break
        ratio_per_node = bytes_to_schedule // (no_nodes - len(scheduled_nodes))
        nodes_to_schedule = set(in_df[in_df.BytesQ <= ratio_per_node].Node.unique()) - scheduled_nodes
        if len(nodes_to_schedule) > 0:
            in_df.loc[in_df.Node.isin(nodes_to_schedule),"Bytes_scheduled"] = in_df[in_df.Node.isin(nodes_to_schedule)]["BytesQ"]
        else:
            nodes_to_schedule = set(in_df[in_df.BytesQ > ratio_per_node].Node.unique()) - scheduled_nodes
            in_df.loc[in_df.Node.isin(nodes_to_schedule),"Bytes_scheduled"] = ratio_per_node
            
        bytes_to_schedule -= in_df[in_df.Node.isin(nodes_to_schedule)]["Bytes_scheduled"].sum()
        scheduled_nodes.update(nodes_to_schedule)
        if len(scheduled_nodes) >= no_nodes:
            assert len(scheduled_nodes) == no_nodes 
            assert in_df["Bytes_scheduled"].sum() <= in_df["Bytes_scheduled_total"].iloc[0]
            break
    return in_df


def apply_scheduling(df, target_mbps):
    if int(df.name) not in target_mbps:
        df["Bytes_scheduled_total"] = df["Bytes"]
        df["BytesQ"] = df["Bytes"]
        return df
    target_bytes = 10**6 * target_mbps[int(df.name)] / (8 * 10)
    df["Bytes_scheduled_total"] = target_bytes
    df["BytesQ"] = df["Bytes"]
    df["Bytes_scheduled"] = 0
    if (df["Bytes"] == 0).all():
        return df
    cross_overs = {}
    last_time = -1
    for g, single_df in df.groupby(["Time"]):
        #single_df = grouped_df.get_group(g)
        
        #Traffic cross-over from one window to the next
        for node in cross_overs:
            node_idx = single_df.Node == node
            if sum(node_idx) > 0:
                single_df.loc[node_idx, "BytesQ"] += cross_overs[node]
        cross_overs = {}
        
        single_df = schedule_bet(single_df)
        cross_over_series = single_df.set_index("Node")["BytesQ"] - single_df.set_index("Node")["Bytes_scheduled"]
        cross_overs = dict(cross_over_series[cross_over_series > 0])
        
        df.loc[single_df.index,:] = single_df   
    
    df["RBs"] = np.ceil(df.Bytes_scheduled.values * 8 / (df.SE.values * 15 * 12))
        
    return df
    

def from_snr_to_traffic(filename, traffic_scenario="standard"):
    f_obj = Path(filename)
    stem = f_obj.stem
    traffic_name = "-".join(stem.split("-")[:-1]) + "-traffic.parquet.gz"
    if traffic_scenario is not None:
        traffic_folder = f"traffic-traces-{traffic_scenario}" 
    else:
        traffic_folder = "traffic_traces"
    traffic_full = str(Path(traffic_folder,traffic_name))
    if path.exists(traffic_full):
        return traffic_full
    else:
        return None

def node_from_name(filename):
    stem = Path(filename).stem
    name_array = stem.split("-")
    if len(name_array) != 8:
        return None
    else:
        return "".join([name_array[2], name_array[4], name_array[5], name_array[6]])
    

def process_monitoring_intervals(
    direction = 'D',
    cell = 0,
    # Aggregation time
    tm = 10,
    # time of interest
    toi = [0, 23200],
    traffic_scenario="standard",
    min_snr = None,
    min_mcs = None,
    max_iter = None,
    cache_df = True,
    cache_cells = True,
    cache_folder = "processed",
    trace_folder = "small-berlin-long-out/trace-*-snr.parquet.gz",
    remove_filter = None,
    custom_name = "",
    static_limit = 5000,
    write_cell_cache=False,
    silent_cache=False,
    target_mbps = {
        'D': {
            2: 500
        },
        'U': {
            2: 2,
            1: 10,
            3: 0.01
        }
    }
):
    assert cache_folder is not None, "POOLERROR"
    filename = f"{cache_folder}/aggregated_{cell}_{direction}_{toi[0]}_{toi[1]}_{tm}_{traffic_scenario}_minmcs_{min_mcs}.parquet"
    if custom_name:
        filename = f"{cache_folder}/aggregated_{cell}_{direction}_{toi[0]}_{toi[1]}_{tm}_{traffic_scenario}_minmcs_{min_mcs}-{custom_name}.parquet"

    if cache_df:
        try:
            df = pd.read_parquet(filename)
            if not silent_cache:
                print(filename, "cached!")
            return df
        except:
            print(filename, "not cached. Processing")
            pass

    if "*" not in trace_folder:
        trace_folder = trace_folder + "/trace-*-snr.parquet.gz"
        
    #cell_json_file = "files_per_cell.json"
    cell_json_file = "files_per_cell.pickle"
    if cache_cells:
        try:
            with open(cell_json_file,'rb') as json_file:
                cells_per_file = pickle.load(json_file)
                copyfile(cell_json_file, cell_json_file + ".bak")
        except FileNotFoundError:
            cells_per_file = {}

    progress_update_size = 0.25
    last_progress_update = 0
    files = glob.glob(trace_folder)
    if remove_filter:
        files = list(filter((lambda x: remove_filter not in x), files))
    total = max_iter if max_iter else len(files)
    last_time = time.time()
    i=-1
    static_count = 0
    aggregate_columns = ["Time", "Node", "Bytes", "SNR", "Slice"]
    aggregate_df = pd.DataFrame(columns=aggregate_columns)
    for snr_file in  files:
        i += 1
        if static_limit is not None and "sta" in snr_file:
            static_no = int(Path(snr_file).name.split("-")[2].split("sta")[1])
            if static_no > static_limit:
                continue
        if cache_cells and snr_file in cells_per_file and cell not in cells_per_file[snr_file]:
            continue
        node = node_from_name(snr_file)
        progress = i / total
        this_progress_update = progress // progress_update_size
        if this_progress_update != last_progress_update:
            this_time = time.time()
            print(f"Progress: {progress*100:.0f}%, elapsed time: {this_time - last_time}")
            last_time = this_time
            last_progress_update = this_progress_update
        if not node:
            continue
        traffic_file = from_snr_to_traffic(snr_file, traffic_scenario)
        try:
            snr_trace = pd.read_parquet(snr_file)
        except EOFError:
            continue
        if cache_cells and snr_file not in cells_per_file:
            cells_per_file[snr_file] = list(snr_trace.Cell.astype(int).unique())
        snr_trace = snr_trace[(toi[0] <= snr_trace.Time) & (snr_trace.Time < toi[1])]
        snr_trace = snr_trace[snr_trace.Cell == cell]
        if direction == 'U':
            snr_trace.SNR = snr_trace.SNR - 21
        snr_trace["Time"] = (snr_trace["Time"] // tm) * tm
        snr_grouped = snr_trace.groupby("Time").agg({'SNR': 'mean'})
        snr_grouped["Node"] = node
        traffic_exists = False
        traffic_slice = None
        if traffic_file:
            try:
                traffic_trace = pd.read_parquet(traffic_file)
            except EOFError:
                continue
            if not traffic_trace.empty: 
                traffic_slice = traffic_trace.Slice.unique()[0]
            traffic_trace = traffic_trace[(toi[0] <= traffic_trace.Time) & (traffic_trace.Time < toi[1])]
            traffic_trace = traffic_trace[traffic_trace.Cell == cell]
            traffic_trace = traffic_trace[traffic_trace.Direction == direction]
            traffic_exists = not traffic_trace.empty
            if traffic_exists:
                traffic_trace["Time"] =  (traffic_trace["Time"] // tm) * tm
                traffic_trace["Bytes"] = np.ceil(traffic_trace["Bytes"])
                traffic_grouped = traffic_trace.groupby(["Time","Slice"]).agg({'Bytes': 'sum'})
                traffic_grouped = traffic_grouped.reset_index().set_index("Time")
                traffic_grouped.index = traffic_grouped.index.astype("float64")
                #print("DBG",snr_grouped.index, traffic_grouped.index,traffic_file,sep="\n")
                last_file = traffic_file
                last_df = traffic_grouped
                grouped = pd.merge(snr_grouped, traffic_grouped, how="left", on="Time")
                grouped = grouped.fillna(0)
                aggregate_df = aggregate_df.append(grouped.reset_index(), ignore_index=True)
        if not traffic_exists:
            grouped = snr_grouped
            grouped['Bytes'] = np.zeros(len(grouped))
            if traffic_slice is None:
                rnd_number = random.uniform(0,1)
                if rnd_number < 0.25:
                    traffic_slice = 4
                else:
                    traffic_slice = 2
            grouped['Slice'] = np.zeros(len(grouped)) + traffic_slice
            aggregate_df = aggregate_df.append(grouped.reset_index(), ignore_index=True)
        if max_iter is not None and i >= max_iter:
            break
    if cache_cells and write_cell_cache:
        with open(cell_json_file, 'wb') as outfile:
            pickle.dump(cells_per_file, outfile)
    # inputs
    if aggregate_df.empty:
      return aggregate_df
    aggregate_df = aggregate_df.sort_values("Time")
    reference_size = 1500
    processed_columns = aggregate_columns + ["SE", "MCS", "RBs"]

    def line_processor(line):
        snr = line.SNR
        mcs = Util.getShannonMCS(snr, mcs_table=BIDIR_MCS[direction], return_nan=True)
        line["MCS"] = mcs
        if mcs is np.nan:
            return line
        se = BIDIR_MCS[direction][mcs]['se']
        line["SE"] = se
        if line.Bytes != 0:
            rbs = int(np.ceil(line.Bytes * 8 / (se * 15 * 12)))
        else:
            rbs = 0
        return line

    extended_df = pd.DataFrame(columns=processed_columns)
    # Fix, if VoIP belongs to eMBB, then turn slice 4 to 2
    extended_df = extended_df.append(aggregate_df)
    extended_df.loc[extended_df.Slice==4,"Slice"] = 2
    extended_df = extended_df.apply(line_processor, axis=1)
    if min_mcs is not None:
        extended_df = extended_df[extended_df.MCS>=min_mcs]
    if target_mbps is not None:
        extended_df = extended_df.groupby("Slice").apply(apply_scheduling, target_mbps[direction]).drop(columns=["Bytes_scheduled_total"])
    else:
        extended_df["Bytes_scheduled"] = extended_df["Bytes"]
        extended_df["BytesQ"] = extended_df["Bytes"]
    
    if not path.exists(cache_folder):
      os.makedirs(cache_folder)

    extended_df.to_parquet(f"{cache_folder}/aggregated_{cell}_{direction}_{toi[0]}_{toi[1]}_{tm}_{traffic_scenario}_minmcs_{min_mcs}.parquet")
    info_obj = {
        'cell': int(cell),
        'direction': direction,
        'start': toi[0],
        'end': toi[1],
        'monitorization_period': tm
    }
    with open(f"{cache_folder}/aggregated_{cell}_{direction}_{toi[0]}_{toi[1]}_{tm}.json", 'w') as outfile:
        json.dump(info_obj, outfile)

    return extended_df


if __name__ == "__main__":
    
    def error_callback(sim_name):
        def inside(e):
            print("Error in", sim_name,e)
            raise e
        return inside
    
    
    #pandarallel.initialize()
    
    np.warnings.filterwarnings('ignore')

    if len(sys.argv) > 1:
      process_monitoring_intervals(cell=0, direction='D', tm=10, traffic_scenario="standard", cache_folder="test", trace_folder='small-berlin-long-out')

    else:
      directions = ['D', 'U']
      tm_and_toi_confs = [(10, (0,23200))]
      scenarios = ["standard", "sporadic", "undistributed"]
      
      cells = range(21)
      with Pool(processes=8) as pool:
          for scenario in scenarios:
              for tm, toi in tm_and_toi_confs:
                  for direction in directions:
                      for cell in cells:
                          gname = f"generation_{direction}_{scenario}_{tm}_{toi}_{cell}"
                          pool.apply_async(
                              process_monitoring_intervals, 
                              kwds={
                                  'cell': cell,
                                  'direction': direction,
                                  'tm': tm,
                                  'toi': toi,
                                  'traffic_scenario': scenario,
                                  'cache_folder': 'aggregated',
                                  'trace_folder': 'small-berlin-long-out'
                              }, 
                              error_callback=error_callback(gname)
                          )
                          print(cell, direction, tm, toi)
          time.sleep(2)
          pool.close()
          pool.join()
