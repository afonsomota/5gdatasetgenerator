import xml.etree.ElementTree as ET
import sys
import pickle
import glob
import numpy as np
import pandas as pd
from operator import itemgetter
from collections import OrderedDict
from ast import literal_eval

from pathlib import Path

from Util_map import Util
from Medium import Simulation, SNRCalculator, UE, ENB, Medium, channel_params, MPFadingModel

class PowerTracker:

  def __init__(self, i='~/sumo_simulations/small-berlin-long/fcd.xml', out="small-berlin-long-out", channel="UMa_A"):
    self.output = out
    self.input = i
    f = open(out + "/sim.config", "rb")
    extract = pickle.load(f)
    f.close()
    self.x_min = extract["x_min"]
    self.y_min = extract["y_min"]
    self.x_max = extract["x_max"]
    self.y_max = extract["y_max"]
    self.max_distance = extract["max_distance"]
    self.sumo_center = extract["sumo_center"]
    self.center = extract["center"]
    self.precision = extract["precision"]
    self.snr_calculator = extract["snr_calculator"]
    f = open(out + "/max_snr_per_pos.pkl", "rb")
    self.max_snr_per_pos = pickle.load(f)
    f.close()
    f = open(out + "/cells.pkl", "rb")
    self.cells = pickle.load(f)
    f.close()
    self.absent_counter = 5
    self.channel = channel
    self.to_write_df = pd.DataFrame()
    self.first_csv_chunk = True

  def parseFile(self, count=0):
    # TODO too many files opened
    f = open(self.input, "r")
    context = ET.iterparse(f, events=("start", "end"))
    root = None
    tstep = 0
    inside = False
    files = {}
    types = {}
    parents = []

    ts_vehicles = None
    last_ts_vehicles = None
    veh_visits = {}

    if count == 0:
      counting = False
    else:
      counting = True

    print("Parsing FCD")
    for event, el in context:
      if root is None:
        root = el
        parents.append(el)
      elif event == "start":
        parent = parents[-1]
        parents.append(el)
      elif event == "end":
        p = parents.pop()
        assert p == el
        if len(parents) > 0:
          parent = parents[-1]
        else:
          assert p == root

      if event == "start" and el.tag == "timestep":
        tstep = float(el.get("time"))
        inside = True
        ts_vehicles = set()
      elif event == "end" and el.tag == "timestep":
        if last_ts_vehicles is not None:
          closed_ts_vehicles = last_ts_vehicles - ts_vehicles
          for v in closed_ts_vehicles:
        #    #            print("closed",v,"in",tstep)
        #    files[v].close()
            files.pop(v)
          last_ts_vehicles.clear()
        last_ts_vehicles = ts_vehicles.copy()
        inside = False
      elif event == "start" and inside:
        if counting:
          count -= 1
          if count == 0:
            break
        t = el.get('type')
        i = el.get('id')
        xa = float(el.get('x')) - self.sumo_center[0]
        ya = float(el.get('y')) - self.sumo_center[1]
        speed = el.get('speed')
        angle = el.get('angle')
        #        if xa < x_min or xa > x_max or ya < y_min or ya > y_max:
        #          continue
        (x, y) = Util.reference_position((xa, ya), self.precision, self.center)
        if x not in self.max_snr_per_pos or y not in self.max_snr_per_pos[x]:
          continue
        out_file = None
        if t is None:
          t = "pedestrian"
        if i not in files:
          if i in veh_visits:
            veh_visits[i] = veh_visits[i] + 1
          else:
            veh_visits[i] = 1
          files[i] = self.output + "/trace-" + str(tstep) + "-" + i + "-" + t + "-v" + str(veh_visits[i]) + ".csv"
          out_file = open(files[i], "w")
        else:
          out_file = open(files[i], "a")
        #          print(i,veh_visits[i])
        ts_vehicles.add(i)
        los = self.max_snr_per_pos[x][y]['los']
        h = self.max_snr_per_pos[x][y]['h']
        cell = self.cells[self.max_snr_per_pos[x][y]['cell']]
        bs = cell['pos']
        lobe = cell['lobe']
        snr = self.snr_calculator.getSNR(bs, (xa, ya, self.snr_calculator.ue.height), lobe, pre_los=los, pre_h=h)
        print(tstep, xa, ya, speed, angle, los, bs, lobe, sep="\t", file=out_file)
        out_file.close()

      if event == "end" and el != root:
        parent.remove(el)

    return self.output + "/trace-*.csv"

  def create_channel(self, angle, speed, los, xa, ya, bs, time_interval, t_s=0.001):
    if not los:
      rho = 0
      sigma_0 = 1 / np.sqrt(2)
    else:
      miu_k = channel_params[self.channel]['K-factor']['miu_k']
      sigma_k = channel_params[self.channel]['K-factor']['sigma_k']
      k = np.random.normal(miu_k, sigma_k**2)
      k_lin = 10**(k/10)
      rho = np.sqrt(k_lin/(k_lin+1))
      sigma_0 = np.sqrt(1/(np.sqrt(2)*(k_lin+1)))
    f_c = channel_params[self.channel]['f_c']
    lambda_0 = 3*10**8 / f_c
    f_max = speed / lambda_0
    ue_bs_angle = np.degrees(np.arctan2(xa-bs[0], ya-bs[1]))
    alpha = np.radians(angle - ue_bs_angle)
    f_d = f_max * np.cos(alpha)
    h_ue = channel_params[self.channel]['h_ue']
    h_bs = channel_params[self.channel]['h_bs']
    (x, y) = Util.reference_position((xa, ya), self.precision, self.center)
    bs_pos = self.cells[self.max_snr_per_pos[x][y]['cell']]['pos']
    d3d = np.sqrt((h_bs-h_ue)**2 + (bs_pos[0]-xa)**2 + (bs_pos[1]-ya)**2)
    theta = - 2*np.pi*d3d/lambda_0
    t_sim = time_interval[1] - time_interval[0]
    # print("fading model",t_sim, f_max, f_c, sigma_0, rho, f_d, theta,time_interval[0], time_interval[1],speed)
    # print("fading model", t_sim)
    fading_model = MPFadingModel(t_sim, f_max, f_c, sigma_0, rho, f_d, theta, t_s=t_s, t_start=time_interval[0])
    time, fading_envelope, mu_i = fading_model.run()

    fe_time = OrderedDict()
    for t in range(len(time)):
      fe_time[time[t]] = fading_envelope[t]

    return fe_time


  # If having memmory issues you can save chunks to CSV (parquet does not allow appends)
  def save_to_csv(self, filename, node_id, sinr_entries, time_values, sumo_ts, t_start, is_final=False):
    #print("Saving to csv", filename)
    entries = []
    columns = ["Time", "X", "Y", "SNR", "Cell", "Node"]
    for k in sinr_entries:
      t_ref = Util.reference_time(k, sumo_ts, t_start)
      entry = [
        k,
        time_values[t_ref]['x'],
        time_values[t_ref]['y'],
        sinr_entries[k] + time_values[t_ref]['snr'],
        time_values[t_ref]['cell'],
        node_id
      ]
      entries.append(entry)
    df = pd.DataFrame(entries, columns=columns).set_index("Time")
    if self.first_csv_chunk:
      df.to_csv(filename, compression='gzip')
      self.first_csv_chunk = False
    else:
      df.to_csv(filename, compression='gzip', header=False, mode='a', index=False)


  def save_to_parquet(self, filename, node_id, sinr_entries, time_values, sumo_ts, t_start, is_final=False):
    #print("Saving to csv", filename)
    entries = []
    columns = ["Time", "X", "Y", "SNR", "Cell", "Node"]
    for k in sinr_entries:
      t_ref = Util.reference_time(k, sumo_ts, t_start)
      entry = [
        k,
        time_values[t_ref]['x'],
        time_values[t_ref]['y'],
        sinr_entries[k] + time_values[t_ref]['snr'],
        time_values[t_ref]['cell'],
        node_id
      ]
      entries.append(entry)
    df = pd.DataFrame(entries, columns=columns)
    self.to_write_df = self.to_write_df.append(df, ignore_index=True)
    if is_final:
      self.to_write_df.to_parquet(filename, compression='gzip')
      self.to_write_df = pd.DataFrame()

  def create_channels(
      self,
      file_name,
      ho=None,
      far_indicator=0.1,
      fading_min=2000,
      fading_max=10000,
      trace_min=30000,
      sumo_ts=0.01,
      fading_ts=0.001,
      sf_corrector=0,
      output_f_name=None
  ):
    main_file = open(file_name, 'r')

    is_far = (lambda a, b: a != b and (b == 0 or abs(a / b - 1) > far_indicator))
    ts_ratio = sumo_ts/fading_ts
    n = 0
    s_o = None
    d_o = None
    x_o = None
    y_o = None
    los_o = None
    bs_o = None
    lobe_o = None
    t_start = None
    changed = False
    angles = []
    speeds = []
    los_states = []
    file_number = 1
    #  output_f_name = (lambda n: ".".join(file_name.split(".")[:-1]) + "-sinr-" + str(n) + ".out")
    file_entries = 0
    sinr_entries = OrderedDict()
    time_values = OrderedDict()
    last_t = None
    if output_f_name is None:
      filename = ".".join(file_name.split(".")[:-1]) + "-snr" + ".parquet.gz"
    else:
      filename = output_f_name
    f_name = Path(file_name).stem
    trace_info = f_name.split("-")
    assert len(trace_info) == 5
    assert trace_info[0] == "trace"
    #t_start = trace_info[1]
    node_id = trace_info[2] + trace_info[4]  # different visits are considered as different nodes

    for line in main_file.readlines():
      line_arr = line.split("\t")
      t = float(line_arr[0])
      x = float(line_arr[1])
      y = float(line_arr[2])
      speed = float(line_arr[3])
      angle = float(line_arr[4])
      los_obj = literal_eval(line_arr[5])
      los = los_obj[0]
      best_bs = eval(line_arr[6])
      best_lobe = int(line_arr[7])
      if t_start is None:
        t_start = t
      else:
        assert round((t-last_t)/sumo_ts) == 1

      t_ref = Util.reference_time(t, sumo_ts, t_start)
      (x_ref, y_ref) = Util.reference_position((x, y), self.precision, self.center)
      snr = self.max_snr_per_pos[x_ref][y_ref]['snr']
      time_values[t_ref] = {
        'x': x,
        'y': y,
        'snr': snr - sf_corrector,
        'angle': angle,
        'speed': speed,
        'los': los,
        'bs': best_bs,
        'lobe': best_lobe,
        'cell': self.max_snr_per_pos[x_ref][y_ref]['cell']
      }

      # TODO Check handover with HO params
      if ho is None:
        if bs_o is None:
          bs_o = best_bs
          lobe_o = best_lobe
          x_o = x
          y_o = y
        elif bs_o != best_bs or lobe_o != best_lobe:
          #print("Changed BS from ", bs_o, lobe_o, "to",best_bs, best_lobe)
          bs_o = best_bs
          lobe_o = best_lobe
          # Create last channel
          if n*10 > fading_min:
            if changed:
              s_o = sum(speeds) / len(speeds)
              d_o = sum(angles) / len(angles)
              los_o = sum(los_states) / len(los_states) >= 0.5
            sinr_entries.update(
              self.create_channel(
                d_o,
                s_o,
                los_o,
                (x+x_o)/2,
                (y+y_o)/2,
                bs_o,
                [t_start, last_t],
                t_s=fading_ts
              )
            )
          # Create new file
          if len(sinr_entries)*(fading_ts/0.001) > trace_min:
            self.save_to_parquet(filename, node_id, sinr_entries, time_values, sumo_ts, t_start)
            # f = open(output_f_name(file_number), "w")
            # print(
            #   "Time",
            #   "Fading",
            #   "Pathloss",
            #   "x",
            #   "y",
            #   "angle",
            #   "speed",
            #   "los",
            #   "bs",
            #   "lobe",
            #   sep="\t",
            #   file=f
            # )
            # for k in sinr_entries:
            #   save_to_parquet(self, filename, node_id, sinr_entries, time_values, sumo_ts, t_start)
            #   t_ref = Util.reference_time(k, sumo_ts, t_start)
            #   print(
            #     k,
            #     sinr_entries[k],
            #     time_values[t_ref]['snr'],
            #     time_values[t_ref]['x'],
            #     time_values[t_ref]['y'],
            #     time_values[t_ref]['angle'],
            #     time_values[t_ref]['speed'],
            #     time_values[t_ref]['los'],
            #     time_values[t_ref]['bs'],
            #     time_values[t_ref]['lobe'],
            #     sep="\t",
            #     file=f
            #   )
            # f.close()
            # file_number += 1
            sinr_entries = {}
            s_o = speed
            d_o = angle
            los_o = los
            x_o = x
            y_o = y
            t_start = t
            speeds = []
            angles = []
            los_states = []
            changed = False
            n = 0
      else:
        assert 1 != 1, "Specified handover procedure not implemented"

      if s_o is None:
        s_o = speed
        d_o = angle
        los_o = los
        t_start = t
      elif is_far(speed, s_o) or is_far(angle, d_o) or los != los_o or changed or n*ts_ratio > fading_max:
        if n*ts_ratio > fading_min:
          if changed:
            s_o = sum(speeds) / len(speeds)
            d_o = sum(angles) / len(angles)
            los_o = sum(los_states) / len(los_states) >= 0.5
          sinr_entries.update(
            self.create_channel(
              d_o,
              s_o,
              los_o,
              (x+x_o)/2,
              (y+y_o)/2,
              bs_o,
              [t_start, last_t],
              t_s=fading_ts
            )
          )
          file_entries += t - t_start
          s_o = speed
          d_o = angle
          los_o = los
          x_o = x
          y_o = y
          t_start = t
          speeds = []
          angles = []
          los_states = []
          changed = False
          n = 0
        else:
          # if not changed:
          #   print("Changed", n, is_far(speed, s_o), is_far(angle, d_o), los != los_o, speed, s_o)
          changed = True

      angles.append(angle)
      speeds.append(speed)
      los_states.append(los)
      n += 1
      last_t = t
      assert len(angles) == n

    if changed:
      s_o = sum(speeds) / len(speeds)
      d_o = sum(angles) / len(angles)
      los_o = sum(los_states) / len(los_states) >= 0.5
    sinr_entries.update(
      self.create_channel(
        d_o,
        s_o,
        los_o,
        (x+x_o)/2,
        (y+y_o)/2,
        bs_o,
        [t_start, t],
        t_s=fading_ts
      )
    )
    # Create new file
    if len(sinr_entries) > trace_min:
      self.save_to_parquet(filename, node_id, sinr_entries, time_values, sumo_ts, t_start, is_final=True)
      # f = open(output_f_name(file_number), "w")
      # print(
      #   "Time",
      #   "Fading",
      #   "Pathloss",
      #   "x",
      #   "y",
      #   "angle",
      #   "speed",
      #   "los",
      #   "bs",
      #   "lobe",
      #   sep="\t",
      #   file=f
      # )
      # for k in sinr_entries:
      #   t_ref = Util.reference_time(k, sumo_ts, t_start)
      #   x_k = time_values[t_ref]['x']
      #   y_k = time_values[t_ref]['y']
      #   snr_k = time_values[t_ref]['snr']
      #   print(
      #     k,
      #     sinr_entries[k],
      #     snr_k,
      #     x_k,
      #     y_k,
      #     time_values[t_ref]['angle'],
      #     time_values[t_ref]['speed'],
      #     time_values[t_ref]['los'],
      #     time_values[t_ref]['bs'],
      #     time_values[t_ref]['lobe'],
      #     sep="\t",
      #     file=f
      #   )
      # f.close()
    main_file.close()


if __name__ == '__main__':
  if len(sys.argv) > 1:
    inp = sys.argv[1]
    out = sys.argv[2]
  else:
    inp = '~/sumo_simulations/small-berlin-long/fcd.xml'
    out = 'small-berlin-long-out'
  if len(sys.argv) > 3:
    min_t = float(sys.argv[3])
    max_t = float(sys.argv[4])
  else:
    min_t = 0
    max_t = None
  maximum_logs = None
  p = PowerTracker(i=inp, out=out)
  trace_files = p.parseFile()
  #trace_files = out + "/trace-*.csv"
  i = 0
  file_number=0
  glob_files = glob.glob(trace_files)
  for f_path in glob_files:
    f_name = Path(f_path).stem
    f_folder = Path(f_path).parent
    trace_info = f_name.split("-")
    start_t = float(trace_info[1])
    file_number+=1
    if max_t is not None and start_t >= max_t or start_t < min_t:
      continue
    print(f"File {file_number} of {len(glob_files)}")
    node_id = trace_info[2] + trace_info[4]  # different visits are considered as different nodes
    node_class = trace_info[3]
    class_keys = []
    if node_class == "pedestrian" or node_class == "bike_bicycle":
      class_keys.append("pedestrian")
    elif node_class == "veh_passenger":
      class_keys.append("vehicle")
      number_of_passengers = np.random.randint(0, 5)
      class_keys += ["passenger"] * number_of_passengers
    else:
      continue
    u_id = 0
    for class_key in class_keys:
      out_f_name = f_name + "-" + class_key + "-" + str(u_id) +  "-snr" + ".parquet.gz"
      print(out_f_name)
      sf = 0
      if class_key == "passenger":
        sf = np.random.normal(9,5)
      p.create_channels(
        f_path,
        fading_min=0,
        trace_min=0,
        sumo_ts=0.1,
        sf_corrector= sf,
        output_f_name=str(Path(f_folder, out_f_name))
      )
      u_id += 1
      i += 1
      if maximum_logs is not None and i >= maximum_logs:
        break
    if maximum_logs is not None and i >= maximum_logs:
      break
