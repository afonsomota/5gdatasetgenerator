import numpy as np
from scipy import stats
from pathlib import Path
import glob
import json
import pickle
import os
from Medium import *
import random
from Util_map import Util
import pandas as pd

import Configurations
from Util import DebuggerStopwatch


class Generator:

  def __init__(self):
    self.size_gen = None
    self.time_gen = None
    self.inter_packet_average = None
    self.main_direction = "downlink"
    assert 1 != 1, "Generator is an abstract class"

  def generate_all(self, size=1, max_t=None):
    if max_t is not None:
      size = int(round(max_t / self.inter_packet_average))
    sizes = self.size_gen(size=size)
    times = self.time_gen(size=size)
    if self.main_direction == "downlink":
      return (np.cumsum(times), sizes), ([], [])
    else:
      return ([], []), (np.cumsum(times), sizes)

  def generate_next(self):
    return self.generate_all(size=1)


class BurstGenerator:

  def __init__(self, vehicle_log, occurrences):
    pass

  def generate_all(self, size=1, max_t=None):
    pass


class MTCSensors(Generator):

  def __init__(self, inter_packet_time=234, packet_size=200, config=None):

    if config is not None:
      inter_packet_time = config.get("inter_packet_time", inter_packet_time)
      packet_size = config.get("packet_size", packet_size)

    def size_generator(size=1):
      def generate_one(idx=0):
        while True:
          generated = stats.norm(packet_size, 0.3 * packet_size).rvs()
          if generated > 10:
            break
        return round(generated)

      if type(size) is int and size == 1:
        return generate_one()
      generate_np = np.vectorize(generate_one)
      if type(size) is int:
        size = (size,)
      return np.fromfunction(generate_np, size, dtype='int32')

    self.size_gen = size_generator
    self.time_gen = stats.expon(inter_packet_time).rvs
    self.inter_packet_average = inter_packet_time
    self.main_direction = "uplink"


class OfflineVideo(Generator):

  def __init__(self, inter_video_session=3600, video_session_duration=20 * 60, config=None):
    self.last_time = 0
    self.times = []
    self.sizes = []

    if config is not None:
      inter_video_session = config.get("inter_video_session", inter_video_session)
      video_session_duration = config.get("video_session_duration", video_session_duration)

    self.inter_video_session = inter_video_session
    self.video_session_duration = video_session_duration

  def generate_all(self, size=1, max_t=None):
    n_videos = 0
    while (
        (max_t is not None and self.last_time < max_t) or
        (max_t is None and n_videos > size)
    ):
      next_interval = np.random.exponential(self.inter_video_session)
      self.last_time += next_interval
      session_duration = np.random.exponential(self.video_session_duration)
      t = 0
      start_stamp = self.last_time
      while t < session_duration:
        last_ts = self.last_time
        self.generate_video()
        n_videos += 1
        t = self.last_time - start_stamp
    return (self.times, self.sizes), ([], [])

  def generate_video(self, video_type=None, conf_dir="youtube/youtube_traces"):
    if video_type is None:
      p = Path(conf_dir, "*.cnf")
      files = glob.glob(str(p))
      f_name = np.random.choice(files)
    else:
      p = Path(conf_dir, video_type + ".cnf")
      f_name = str(p)
    cnf_file = open(f_name)
    cnf = json.load(cnf_file)
    param = cnf['inter_packet_distribution']['params']
    dist = cnf['inter_packet_distribution']['distribution']
    packet_list = cnf['packets_sizes']
    no_packets = len(packet_list)
    class_duration = cnf['video_length']
    inter_packet_generator = getattr(stats, dist)(*param[:-2], loc=param[-2], scale=param[-1]).rvs
    while True:
      clip_duration = stats.norm(class_duration, 0.3 * class_duration).rvs()
      if clip_duration > 0:
        break
    clip_no_packets = no_packets * int(round(clip_duration / class_duration))
    idx = 0
    start_time = self.last_time
    for n in range(clip_no_packets):
      # circular iteration of the last 2/3 to avoid repeating the behavior in the begining of the video
      if idx == no_packets:
        idx = round(no_packets * 1 / 3)
      original_packet = packet_list[idx]
      while True:
        gen_packet = stats.norm(original_packet, 0.3 * original_packet).rvs()
        if gen_packet > 0:
          break
      self.sizes.append(gen_packet)
      self.times.append(self.last_time)
      if n == clip_no_packets:
        self.last_time = start_time + clip_duration
      else:
        self.last_time += inter_packet_generator()


class Bursts:

  def __init__(
      self,
      sim_config="../sinr-map/small-berlin-out/sim.config",
      total_burst_area=None,
      inter_burst=60,
      burst_radius=50
  ):
    self.burst_time_generator = stats.expon(inter_burst).rvs
    assert sim_config is not None or total_burst_area is not None
    if sim_config is None:
      self.total_burst_area = total_burst_area  # ((x_min, x_max), (y_min, y_max))
    else:
      with open(sim_config, "rb") as conf_file:
        medium_conf = pickle.load(conf_file)
        self.total_burst_area = (
          (medium_conf['x_min'], medium_conf['x_max']),
          (medium_conf['y_min'], medium_conf['y_max'])
        )
    self.burst_radius = burst_radius
    self.last_time = 0
    self.bursts = []  # (center, radius)

  def generate_burst(self):
    self.last_time += self.burst_time_generator()
    # self.times.append(self.last_time)
    x_center = stats.randint(
      self.total_burst_area[0][0] + self.burst_radius,
      self.total_burst_area[0][1] - self.burst_radius,
    ).rvs()
    y_center = stats.randint(
      self.total_burst_area[1][0] + self.burst_radius,
      self.total_burst_area[1][1] - self.burst_radius,
    ).rvs()
    # self.areas.append({'center': (x_center, y_center), 'radius': self.burst_radius})
    self.bursts.append({'time': self.last_time, 'center': (x_center, y_center), 'radius': self.burst_radius})

  def generate_all(self, size=1, max_t=None):
    n_bursts = 0
    while (
        (max_t is not None and self.last_time < max_t) or
        (max_t is None and n_bursts > size)
    ):
      self.generate_burst()
      n_bursts += 1
    return self.bursts


class V2I(Generator):

  def __init__(self, config=None):
    miu = 400
    sigma = 200
    min_val = 200
    max_val = 500
    self.size_gen = stats.truncnorm((min_val - miu) / sigma, (max_val - miu) / sigma, loc=miu, scale=sigma).rvs
    miu = 0.4
    sigma = 1 / (0.4 * np.sqrt(2 * np.pi))  # p(X=miu) = 0.4
    min_val = 0.1
    max_val = 1
    self.time_gen = stats.truncnorm((min_val - miu) / sigma, (max_val - miu) / sigma, loc=miu, scale=sigma).rvs
    self.inter_packet_average = 0.4
    self.main_direction = "uplink"
    # if ssm_file_name is None:
    #  self.burst_times = []
    # else:
    #  assert vehicle_id is not None
    #  ssm_xml_tree = ET.parse(ssm_file_name)
    #  root = ssm_xml_tree.getroot()


# class V2IBurst(BurstGenerator):
#
#   def __init__(self, vehicle_log, occurrences, burst_size=320, config=None):
#     self.occurrences = iter(occurrences)
#     self.log_file = open(vehicle_log)
#
#     if config is not None:
#       burst_size = config.get("burst_size", burst_size)
#
#     self.burst_size = burst_size
#     self.times = []
#     self.sizes = []
#
#   def generate_all(self, size=1, max_t=None):
#     n_occurrences = 0
#     log_iterator = iter(self.log_file.readlines())
#     current_log_entry = next(log_iterator)
#
#     def log_time(line):
#       try:
#         return float(line.split("\t", 1)[0])
#       except ValueError:
#         return 0
#
#     t = 0
#     while (
#         (max_t is not None and t < max_t) or
#         (max_t is None and n_occurrences > size)
#     ):
#       try:
#         current_occurrence = next(self.occurrences)
#       except StopIteration:
#         break
#       t = current_occurrence['time']
#       center = current_occurrence['center']
#       radius = current_occurrence['radius']
#       while current_log_entry is not None and log_time(current_log_entry) < t:
#         try:
#           current_log_entry = next(log_iterator)
#         except StopIteration:
#           break
#       log_entry_arr = current_log_entry.split("\t")
#       pos = (float(log_entry_arr[1]), float(log_entry_arr[2]))
#       angle = float(log_entry_arr[3])
#       distance = np.linalg.norm(np.array(pos) - np.array(center))
#       # TODO is vehicle going towards or away form incident
#       if distance < radius:
#         self.times.append(t)
#         self.sizes.append(self.burst_size)
#       n_occurrences += 1
#     return ([], []), (self.times, self.sizes)


class V2IBurst(BurstGenerator):

  def __init__(self, vehicle_df, occurrences, burst_size=320, config=None):
    self.occurrences = iter(occurrences)
    self.vehicle_df = vehicle_df

    if config is not None:
      burst_size = config.get("burst_size", burst_size)

    self.burst_size = burst_size
    self.times = []
    self.sizes = []

  def generate_all(self, size=1, max_t=None):
    n_occurrences = 0

    t = 0
    while (
        (max_t is not None and t < max_t) or
        (max_t is None and n_occurrences > size)
    ):
      try:
        current_occurrence = next(self.occurrences)
      except StopIteration:
        break
      t = current_occurrence['time']
      center = current_occurrence['center']
      radius = current_occurrence['radius']
      idx = self.vehicle_df.Time.searchsorted(t, 'right') - 1
      pos = (self.vehicle_df.iloc[idx].X, self.vehicle_df.iloc[idx].Y)
      self.vehicle_df = self.vehicle_df.iloc[idx:]
      distance = np.linalg.norm(np.array(pos) - np.array(center))
      if distance < radius:
        self.times.append(t)
        self.sizes.append(self.burst_size)
      n_occurrences += 1
    return ([], []), (self.times, self.sizes)


class VoIP(Generator):

  def __init__(self, inter_call=10 * 60, average_call_duration=3 * 60, config=None):
    self.sizes_up = []
    self.times_up = []
    self.sizes_down = []
    self.times_down = []

    self.p_talking = 0.01
    self.p_silent = 0.01
    self.T = 0.02
    self.talking_size = 40
    self.sid_size = 15
    self.is_talking = True
    self.silent_threshold = 0.16
    self.last_time = 0

    if config is not None:
      inter_call = config.get("inter_call", inter_call)
      average_call_duration = config.get("average_call_duration")

    self.inter_call = inter_call
    self.average_call_duration = average_call_duration

  def generate_call(self, duration, last_time, times, sizes):
    call_time = 0
    silent_duration = 0
    is_talking = True
    while call_time < duration:
      if is_talking:
        silent_duration = 0
        call_time += self.T
        last_time += self.T
        sizes.append(self.talking_size)
        times.append(last_time)
        if np.random.uniform() <= self.p_silent:
          is_talking = False
      else:
        silent_duration += self.T
        call_time += self.T
        last_time += self.T
        if silent_duration % self.silent_threshold == 0:
          sizes.append(self.sid_size)
          times.append(last_time)
        if np.random.uniform() <= self.p_talking:
          is_talking = True
    return last_time

  def generate_all(self, size=1, max_t=None):
    n_calls = 0
    while (
        (max_t is not None and self.last_time < max_t) or
        (max_t is None and n_calls > size)
    ):
      next_interval = np.random.exponential(self.inter_call)
      self.last_time += next_interval
      call_duration = np.random.exponential(self.average_call_duration)
      self.generate_call(call_duration, self.last_time, self.times_up, self.sizes_up)
      self.last_time = self.generate_call(call_duration, self.last_time, self.times_down, self.sizes_down)
    return (self.times_down, self.sizes_down), (self.times_up, self.sizes_up)


class Web(Generator):

  def __init__(
      self,
      initial_request=None,
      inter_session=10 * 60,
      pages_per_session=3.86,
      pages_per_session_std=2.465,
      config=None
  ):
    self.sizes = []
    self.times = []

    if config is not None:
      inter_session = config.get("inter_session", inter_session)
      pages_per_session = config.get("pages_per_session", pages_per_session)
      pages_per_session_std = config.get("pages_per_session_std", pages_per_session_std)

    if initial_request is None:
      self.last_time = np.random.exponential(1 / 0.033)
    else:
      self.last_time = 0
    self.main_obj_size = FakeTruncatedLogNormalGenerator(100, 2 * 10 ** 6, 8.37, 1.37)
    self.obj_size = FakeTruncatedLogNormalGenerator(50, 2 * 10 ** 6, 2.36, 6.17)
    self.n_objs = FakeTruncatedParetoGenerator(0, 53, 1.11, 2)
    self.reading_time = AbstractGenerator(np.random.exponential, 1 / 0.033)
    self.parsing_time = AbstractGenerator(np.random.exponential, 1 / 7.69)
    self.inter_session_gen = stats.expon(inter_session).rvs

    def pages_per_session_gen():
      while True:
        generated = int(round(stats.norm(pages_per_session, pages_per_session_std).rvs()))
        if generated > 0:
          break
      return generated

    self.pages_per_session_gen = pages_per_session_gen

  def generate_all(self, size=1, max_t=None):
    n_pages = 0
    while (
        (max_t is not None and self.last_time < max_t) or
        (max_t is None and n_pages > size)
    ):
      next_section_diff = self.inter_session_gen()
      self.last_time += next_section_diff
      session_no_pages = self.pages_per_session_gen()
      for i in range(session_no_pages):
        self.generate_page()
        n_pages += 1
    return (self.times, self.sizes), ([], [])

  def generate_page(self):
    # main page
    self.times.append(self.last_time)
    self.sizes.append(round(self.main_obj_size.generate()))
    # parse file
    self.last_time += self.parsing_time.generate()
    # download objects
    n_objs = round(self.n_objs.generate())
    # print("No Objs", n_objs)
    if n_objs != 0:
      total_size = self.obj_size.generate((n_objs,)).round().sum()
      self.times.append(self.last_time)
      self.sizes.append(total_size)
    # reading the content
    self.last_time += self.reading_time.generate()


class FakeTruncatedLogNormalGenerator:

  def __init__(self, min_val, max_val, miu, sigma):
    self.min_val = min_val
    self.max_val = max_val
    self.miu = miu
    self.sigma = sigma

  def generate(self, size=1):
    def generate_one(idx=0):
      while True:
        gen_val = np.random.lognormal(self.miu, self.sigma)
        if self.min_val <= gen_val <= self.max_val:
          break
      return gen_val

    if type(size) is int and size == 1:
      return generate_one()
    elif type(size) is int and size == 0:
      return None
    generate_np = np.vectorize(generate_one)
    if type(size) is int:
      size = (size,)
    return np.fromfunction(generate_np, size)


class FakeTruncatedParetoGenerator:

  def __init__(self, min_val, max_val, alpha, k):
    self.min_val = min_val
    self.max_val = max_val
    self.alpha = alpha
    self.k = k

  def generate(self, size=1):
    def generate_one(idx=0):
      while True:
        gen_val = round((1 + np.random.pareto(self.alpha)) * self.k - self.k)
        if self.min_val <= gen_val <= self.max_val:
          break
      return gen_val

    if type(size) is int and size == 1:
      return generate_one()
    generate_np = np.vectorize(generate_one)
    if type(size) is int:
      size = (size,)
    return np.fromfunction(generate_np, size)


class AbstractGenerator:

  def __init__(self, generator, *args):
    self.generator = generator
    self.args = args

  def generate(self, size=1):
    if type(size) is int and size == 1:
      return self.generator(*self.args)
    if type(size) is int:
      size = (size,)
    return self.generator(*self.args, size=size)


def choose_from_prob_array(array, prob_key="probability", value_key="model"):
  if type(array) is str:
    print("Not array:", array)
  weights = map((lambda x: x[prob_key]), array)
  if type(value_key) is str:
    values = list(map((lambda x: x[value_key]), array))
  else:
    values_list = []
    for k in value_key:
      vals = list(map((lambda x: x[k]), array))
      values_list.append(vals)
    values = list(zip(*values_list))

  return random.choices(values, weights)[0]


def trim_excess(times_and_sizes, timeout):
  (times_down, sizes_down), (times_up, sizes_up) = times_and_sizes
  times_down = np.array(times_down)
  sizes_down = np.array(sizes_down)
  times_up = np.array(times_up)
  sizes_up = np.array(sizes_up)
  filter_down = times_down <= timeout
  filter_up = times_up <= timeout
  return (times_down[filter_down], sizes_down[filter_down]), (times_up[filter_up], sizes_up[filter_up])


def generate_traffic(config, traffic_types, occurrences=None, vehicle_df=None, t_start=0,  timeout=None):
  total_times_down = []
  total_sizes_down = []
  total_times_up = []
  total_sizes_up = []
  if timeout is None:
    timeout = config['timeout']
  for ttype in traffic_types:
    type_obj = config["traffic"][ttype]
    traffic_config = choose_from_prob_array(type_obj['config'], value_key="value")
    if type_obj['class'].__name__ == "V2IBurst":
      if occurrences is None:
        continue
      traffic_instance = type_obj['class'](vehicle_df, occurrences, config=traffic_config)
    else:
      traffic_instance = type_obj['class'](config=traffic_config)
    (times_down, sizes_down), (times_up, sizes_up) = trim_excess(traffic_instance.generate_all(max_t=timeout), timeout)
    total_times_down += list(np.array(times_down) + t_start)
    total_sizes_down += list(np.array(sizes_down))
    total_times_up += list(np.array(times_up) + t_start)
    total_sizes_up += list(np.array(sizes_up))

  return (total_times_down, total_sizes_down), (total_times_up, total_sizes_up)


def generate_data(config, only_static_snr=False):
  sw = DebuggerStopwatch()
  scene_conf = pickle.load(open(config["scene_conf_file"], "rb"))
  total_area = (scene_conf["x_max"] - scene_conf["x_min"]) * (scene_conf["y_max"] - scene_conf["y_min"])
  snr_per_pos = pickle.load(open(config["snr_file"], "rb"))
  number_of_static = round(config["static"]["density"] * total_area)
  if config["static"].get("inside") is not None:
    inside_nodes = round(config["static"].get("inside") * number_of_static)
    outside_nodes = number_of_static - inside_nodes
  traffic_columns = ["Time", "Node", "Cell", "Bytes", "Direction", "Types"]
  if not os.path.exists(config["output_folder"]):
    os.makedirs(config["output_folder"])
  static_positions_df = pd.DataFrame(columns=["Node", "X", "Y", "SNR"])
  # static_snr_df = pd.DataFrame(columns=["Time", "X", "Y", "SNR", "Cell", "Node"])
  print("Number of static", number_of_static)
  for i in range(number_of_static):
    sw.tick("static-snr-whole")
    node_id = f"sta{i}v1"
    filename = f"trace-0.0-sta{i}-static-v1-static-0-snr.parquet.gz"
    output_path = str(Path(config["trace_folder"], filename))
    if os.path.exists(output_path) and not config.get("overwrite", True):
      continue
    while True:
      x = scene_conf["x_min"] + stats.uniform().rvs() * (scene_conf["x_max"] - scene_conf["x_min"])
      y = scene_conf["y_min"] + stats.uniform().rvs() * (scene_conf["y_max"] - scene_conf["y_min"])
      (x_ref, y_ref) = Util.reference_position(
        (x, y),
        scene_conf['precision'],
        scene_conf['center']
      )
      if x_ref in snr_per_pos and y_ref in snr_per_pos[x_ref]:
        if config["static"].get("inside") is not None:
          is_inside = snr_per_pos[x_ref][y_ref]['los'][1]
          if is_inside and inside_nodes > 0:
            inside_nodes -= 1
          elif is_inside and outside_nodes > 0:
            outside_nodes -= 1
          else:
            continue
        break
    new_static_node = {'x': x, 'y': y, "snr_obj": snr_per_pos[x_ref][y_ref]}

    sw.tick("static-snr-processing")
    times = np.arange(0, config["timeout"], config["sampling_period"])
    snr_lin = 10 ** (new_static_node['snr_obj']['snr']/10)
    error_lin = snr_lin * config["snr_error_std"]
    snr_values = 10 * np.log10(stats.norm(snr_lin, error_lin).rvs(size=len(times)))
    snr_df = pd.DataFrame({
      'Time': times,
      'X': [x] * len(times),
      'Y': [y] * len(times),
      'SNR': snr_values,
      'Cell': [new_static_node['snr_obj']['cell']] * len(times),
      'Node': [node_id] * len(times)
    })
    sw.tick("static-snr-processing")
    static_positions_df = static_positions_df.append({
      'Node': node_id,
      'X': x,
      'Y': y,
      'Cell': new_static_node['snr_obj']['cell'],
      'SNR': new_static_node['snr_obj']['snr']
    }, ignore_index=True)
    compression = 'gzip'

    #sw.tick("static-snr-writing")
    #snr_df.to_csv(output_path, compression=compression, index=False)
    #sw.tick("static-snr-writing")
    sw.tick("static-snr-writing-parquet")
    filename = f"trace-0.0-sta{i}-static-v1-static-0-snr.parquet.gz"
    output_path = str(Path(config["trace_folder"], filename))
    snr_df.to_parquet(output_path, compression=compression, index=False)
    sw.tick("static-snr-writing-parquet")
    sw.tick("static-snr-whole")
    print(i, sw.show_and_clear())

  compression = 'gzip'
  static_positions_df.to_parquet(Path(config["trace_folder"], "static_positions.parquet"), compression=compression, index=False)

  if only_static_snr:
    return

  sw = DebuggerStopwatch()
  parquet_snr_files = glob.glob(str(Path(config["trace_folder"], "trace-0.0-sta*-snr.parquet.gz")))
  # static_positions_df = pd.read_parquet(Path(config["trace_folder"], "static_positions.parquet"))
  static_positions_df = static_positions_df.set_index('Node')
  i = 0
  for f_path in parquet_snr_files:
    output_filename = f"trace-0.0-sta{i}-static-v1-static-0-traffic.parquet.gz"
    output_path = str(Path(config["output_folder"], output_filename))
    i+=1
    if os.path.exists(output_path) and not config.get("overwrite", True):
      continue
    sw.tick("static-traffic-whole")
    trace_info = Path(f_path).name.split("-")
    node_id = trace_info[2] + trace_info[4]
    static_traffic_df = pd.DataFrame(columns=traffic_columns)
    traffic_types, slic = choose_from_prob_array(config['attribution']['static'], value_key=["model", "slice"])
    sw.tick("static-traffic-processing")
    (total_times_down, total_sizes_down), (total_times_up, total_sizes_up) = generate_traffic(config, traffic_types)
    sw.tick("static-traffic-processing")
    sw.tick("static-traffic-filling")
    cell = static_positions_df.loc[node_id,"Cell"]
    static_traffic_df = static_traffic_df.append(pd.DataFrame(
      {
        'Time': total_times_down,
        'Node': [node_id] * len(total_times_down),
        'Cell': [cell] * len(total_times_down),
        'Bytes': total_sizes_down,
        'Direction': ["D"] * len(total_times_down),
        'Types': [traffic_types] * len(total_times_down),
        'Slice': [slic] * len(total_times_down)
      }
    ))
    static_traffic_df = static_traffic_df.append(pd.DataFrame(
      {
        'Time': total_times_up,
        'Node': [node_id] * len(total_times_up),
        'Cell': [cell] * len(total_times_up),
        'Bytes': total_sizes_up,
        'Direction': ["U"] * len(total_times_up),
        'Types': [traffic_types] * len(total_times_up),
        'Slice': [slic] * len(total_times_up)
      }
    ))
    sw.tick("static-traffic-filling")

    sw.tick("static-traffic-writing")
    static_traffic_df.to_parquet(output_path, compression='gzip', index=False)
    sw.tick("static-traffic-writing")
    sw.tick("static-traffic-whole")
    print(i, node_id, traffic_types, sw.show_and_clear())

  bursts = Bursts(sim_config=config["scene_conf_file"], inter_burst=config["bursts"]["inter_burst"],
                  burst_radius=config["bursts"]["radius"]).generate_all(max_t=config["timeout"])

  snr_counter = 0
  parquet_snr_files = glob.glob(str(Path(config["trace_folder"], "trace*-snr.parquet.gz")))
  for f_path in parquet_snr_files:
    snr_counter += 1
    f_name = Path(f_path).stem
    if "-sta" in f_name:
        continue
    try:
      trace_df = pd.read_parquet(f_path)
    except EOFError:
      print("Corrupted file", f_path)
      sw.clear()
      continue
    if snr_counter % 100 == 0:
        print(f"Progress: {100*snr_counter/len(parquet_snr_files):.2f}")
    print("File", f_name)
    trace_info = f_name.split("-")
    assert len(trace_info) == 6 or len(trace_info) == 8
    assert trace_info[0] == "trace"
    t_start = float(trace_info[1])
    if config.get("trace_filter", None) is not None:
        if t_start < config["trace_filter"][0] or t_start >= config["trace_filter"][1]:
            continue
    if len(trace_info) == 6:
      node_id = trace_info[2] + trace_info[4]  # different visits are considered as different nodes
    elif len(trace_info) == 8:
      node_id = trace_info[2] + trace_info[4] + trace_info[5] + trace_info[6]
    else:
      assert 1 != 1, "Incompatible filename for feature extraction."
    node_class = trace_info[3]
    class_keys = []

    is_df_sorted = trace_df["Time"].is_monotonic
    if not is_df_sorted:
      trace_df = trace_df.sort_values("Time")
    t_end = trace_df.iloc[-1]["Time"]

    if len(trace_info) == 6:
      if node_class == "pedestrian":
        class_keys.append("pedestrian")
      elif node_class == "veh_passenger":
        class_keys.append("vehicle")
        number_of_passengers = np.random.randint(0, 5)
        class_keys += ["passenger"] * number_of_passengers
      else:
        continue
      u_id = 0
    else:
      class_keys.append(trace_info[5])
      u_id = trace_info[6]

    for class_key in class_keys:
      if len(trace_info) == 6:
        traffic_f_name = "-".join(f_name.split("-")[:-1] + [str(u_id), "traffic"])
      else:
        traffic_f_name = "-".join(f_name.split("-")[:-1] + ["traffic"])
      traffic_f_name += ".parquet.gz"
      traffic_types, slic = choose_from_prob_array(config['attribution'][class_key], value_key=["model", "slice"])
      output_path = str(Path(config["output_folder"], traffic_f_name))
      if os.path.exists(output_path) and not config.get("overwrite", True):
        print("Traffic types", traffic_types, class_key, "skiped!")
        continue
      else:
        print("Traffic types", traffic_types, class_key)
      (total_times_down, total_sizes_down), (total_times_up, total_sizes_up) = generate_traffic(
        config,
        traffic_types,
        occurrences=bursts,
        vehicle_df=trace_df,
        t_start=t_start,
        timeout=t_end-t_start
      )
      output_df = pd.DataFrame({
        'Time': [],
        'Node': [],
        'Cell': [],
        'Bytes': [],
        'Direction': [],
        'Types': [],
        'Slice': []
      })
      trace_pointer = 0
      for tup in ((total_times_down, total_sizes_down, 'D'), (total_times_up, total_sizes_up, 'U')):
        times = tup[0]
        if not times:
          continue
        sizes = tup[1]
        direction = tup[2]
        last_time = times[0]
        cells = np.zeros(len(times))
        iter_df = trace_df
        for i, t in enumerate(times):
          if t < last_time:
           iter_df = trace_df
          idx = max(iter_df.Time.searchsorted(t, 'right') - 1, 0)
          assert idx >= 0, f"IDX<0 - idx: {idx}, t: {t}, first_iter: {iter_df.iloc[0].Time}, first_total: {trace_df.iloc[0].Time},"
          last_time = t
          cells[i] = iter_df.iloc[idx].Cell
          iter_df = iter_df.iloc[idx:]
        output_df = output_df.append(pd.DataFrame({
            'Time': times,
            'Node': [node_id] * len(times),
            'Cell': cells,
            'Bytes': sizes,
            'Direction': [direction] * len(times),
            'Types': [traffic_types] * len(times),
            'Slice': [slic] * len(times)
        }), ignore_index=True)
        #if is_first[class_key]:
        #  output_df.to_csv(csv_name[class_key], header=True, compression='gzip')
        #else:
        #  output_df.to_csv(csv_name[class_key], header=False, compression='gzip', mode='a')
        #is_first[class_key] = False
      is_df_sorted = (np.diff(output_df["Time"]) > 0).all()
      if not is_df_sorted:
        output_df = output_df.sort_values("Time")
      if not output_df.empty:
        output_df.to_parquet(str(Path(config["output_folder"], traffic_f_name)),
                         compression='gzip', index=False)
      if len(trace_info) == 6:
        u_id += 1
    sw.tick("traffic-whole")


if __name__ == '__main__':
  chosen_mode = None
  start_ts_min = None
  start_ts_max = None
  only_static_snr = False 
  if len(sys.argv) > 1:
    chosen_mode = sys.argv[1]
    assert chosen_mode in Configurations.scenarios, "Unresgistered scenario."
  if len(sys.argv) > 2:
    start_ts_min = int(sys.argv[2])
    start_ts_max = int(sys.argv[3])
  is_first = True
  for mode in Configurations.scenarios:
    if chosen_mode is not None and mode != chosen_mode:
      continue
    print("Mode:", mode)
    gen_configuration = Configurations.get_default_configuration(mode=mode, static=0.00001, out_folder=f"traffic-traces-{mode}")
    if start_ts_min is not None:
      gen_configuration["trace_filter"] = (start_ts_min, start_ts_max)
    gen_configuration["overwrite"] = True
#    gen_configuration["overwrite"] = is_first
#    is_first = False
    generate_data(gen_configuration, only_static_snr=only_static_snr)
    if only_static_snr:
        break
