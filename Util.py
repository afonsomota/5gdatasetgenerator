from fractions import Fraction

from NRTables import MCS, TBS, PC_TBS, MCS2CQI, PC_TBS_short
from scipy.stats import norm
import math, statistics
import numpy as np
import pandas as pd
import time

from functools import reduce

import Medium


# used to generate PC_TBS
def getTBSSize(mcs, rbs, nred=12 * 14, layers=1, dmrs=12):
  npre = nred - dmrs
  nre = min(npre, 156) * rbs
  r = MCS[mcs]['R']
  qm = MCS[mcs]['Qm']
  ninfo = nre * (r / 1024) * qm * layers
  if ninfo <= 3824:
    kb = 3824
    try:
      n = max(3, int(math.log2(ninfo)) - 6)
    except ValueError:
      n = 3
    npinfo = max(24, pow(2, n) * int(ninfo / pow(2, n)))
    _, tbs = treeClosestGE(npinfo, TBS)
    no_cbs = 1
  else:
    kb = 8424
    n = int(math.log2(ninfo - 24)) - 5
    npinfo = max(3840, pow(2, n) * round((ninfo - 24) / pow(2, n)))
    if r / 1024 < 0.25:
      c = math.ceil((npinfo + 24) / 3816)
      tbs = 8 * c * math.ceil((npinfo + 24) / (8 * c)) - 24
      no_cbs = c
    elif npinfo > 8424:
      c = math.ceil((npinfo + 24) / 8424)
      tbs = 8 * c * math.ceil((npinfo + 24) / (8 * c)) - 24
      no_cbs = c
    else:
      tbs = 8 * math.ceil((npinfo + 24) / 8) - 24
      no_cbs = 1

  return tbs, no_cbs


# the use of tracemalloc tool revealed that deepcopies were not deleted
def deep_del(obj):
  if type(obj) == dict:
    keys = list(obj.keys())
    for k in keys:
      deep_del(obj[k])
    del obj


def copy_user_info(ue_traffic):
  new_obj = {}
  for u in ue_traffic:
    new_obj[u] = ue_traffic[u].copy()
  return new_obj


def get_cbs(mcs, rbs):
  A = PC_TBS[mcs][rbs - 1]
  r = MCS[mcs]['R'] / 1024
  if A <= 292 or (A <= 3824 and r <= 0.67) or r <= 0.25:
    bg = 2
    K_cb = 3840
  else:
    bg = 1
    K_cb = 8448

  # plus global CRC
  B = A + 24

  # calculating C
  if B <= K_cb:
    L = 0
    C = 1
    Bp = B
  else:
    L = 24
    C = np.ceil(B / (K_cb - L))
    Bp = B + C * L

  Kp = Bp / C
  if bg == 1:
    K_b = 22
  else:
    if B > 640:
      K_b = 10
    elif B > 560:
      K_b = 9
    elif B > 192:
      K_b = 8
    else:
      K_b = 6

  min_lifting_sizes = np.zeros(8)
  min_lifting_sizes[0] = 2
  min_lifting_sizes[1:] = np.arange(3, 16, 2)
  max_lifting_size = 384

  Z_c = None
  iLS = None

  for i in range(0, len(min_lifting_sizes)):
    zc = min_lifting_sizes[i]
    while zc <= max_lifting_size:
      if K_b * zc >= Kp and (Z_c == None or zc < Z_c):
        Z_c = int(zc)
        iLS = i
      zc *= 2

  if bg == 1:
    K = 22 * Z_c
  else:
    K = 10 * Z_c

  return K, r, bg, iLS, Z_c, C


def getSDUHeader(size):
  assert size <= 65535
  if size <= 255:
    return 2
  else:
    return 3


def getSDUPayload(size):
  assert size != 258
  if size <= 257:
    return size - 2
  else:
    return size - 3


def get_required_rbs(mcs, buf_size, tti):
  amount = buf_size * 8
  pc = get_pc(tti)
  if amount == 0:
    return 0
  if amount > pc[mcs][-1]:
    return len(pc[mcs])
  rb, _ = treeClosestGE(amount, pc[mcs])
  rb = rb + 1  # PC_TBS[0] means 1 RB
  return rb


def get_tbs(mcs, rbs, tti):
  pc = get_pc(tti)
  assert mcs in pc and rbs <= len(pc[mcs])
  return pc[mcs][rbs-1]


def get_pc(tti=0.001):
  pc = None
  if tti == 0.001:
    pc = PC_TBS
  elif tti < 0.0002:
    pc = PC_TBS_short
  assert pc is not None, "TTI not supported"
  return pc


def treeClosestGE(tbs, arr, base=0):
  mid = len(arr) // 2
  if len(arr) == 1:
    return base, arr[0]
  if arr[mid] >= tbs:
    if arr[mid - 1] < tbs:
      return base + mid, arr[mid]
    return treeClosestGE(tbs, arr[0:mid], base)
  else:
    return treeClosestGE(tbs, arr[mid + 1:], base + mid + 1)


def getRXProbability(snr, mcs, tbs, symbol_time=10 ** -3 / 14, bandwidth=15 * 10 ** 3, fec=0):
  # Adapted from Matlab scheduler
  # return min(0.6+MCS2CQI[mcs]/30,0.95)

  # Theoretical AWGN for modulations used
  # l_snr = pow(10,snr/10)
  l_snr = 10 ** (snr / 10)
  # es_n0 = symbol_time * bandwidth * l_snr
  es_n0 = l_snr
  bps = MCS[mcs]['Qm']
  eb_n0 = es_n0 / bps
  # eb_n0 = es_n0
  if bps == 2:
    ber = norm.sf(np.sqrt(2 * eb_n0))
  else:
    ber = (4 / bps) * norm.sf(np.sqrt((3 * eb_n0 * bps) / (2 ** bps - 1)))
    # ber = ((4 * (np.sqrt(2**bps) - 1)) / ((np.sqrt(2**bps) * bps)) * norm.sf(np.sqrt((3 * eb_n0 * bps) / (2 ** bps - 1))))
  return (1 - ber) ** tbs


def getShannonRxProbability(snr, mcs, rbs):
  K, _, _, _, _, C = get_cbs(mcs, rbs)
  snr_lin = 10 ** (snr / 10)
  exponent = -1.5 * snr_lin / (2 ** MCS[mcs]['se'] - 1)
  ber = np.exp(exponent) / 5
  bler = 1 - (1 - ber) ** K
  return (1 - bler) ** C


# cache for MCS calculation
mcs_cache = {}


def getShannonMCS(snr, ber=0.00005, rbs=4, bler=None, mcs_table=MCS, adjustment=1.5, return_nan=False):
  tpl = (snr, ber, rbs, bler, mcs_table[0]['se'], mcs_table[27]['se'], adjustment, return_nan)
  if tpl in mcs_cache:
    return mcs_cache[tpl]
  mcs_ber = np.repeat(ber, len(mcs_table))
  if bler is not None:
    for m in mcs_table:
      if m > 27:
        break
      K, _, _, _, _, C = get_cbs(m, rbs)
      mcs_ber[m] = 1 - (1 - bler) ** K

  phy = - np.log(5 * mcs_ber) / adjustment
  shn = np.log2(1 + 10 ** (snr / 10) / phy)
  if return_nan:
    chosen_mcs = np.nan
  else:
    chosen_mcs = 0
  for m in mcs_table:
    if m > 27:
      break
    se = mcs_table[m]['se']
    shn_m = shn[m]
    if se < shn_m:
      chosen_mcs = m
    else:
      break

  mcs_cache[tpl] = chosen_mcs
  return chosen_mcs


def getMeanCIPair(list):
  if len(list) == 0:
    return (0, 0)
  avg = statistics.mean(list)
  std = statistics.pstdev(list)
  lnt = len(list)
  return (avg, 1.96 * std / math.sqrt(lnt))


def orderLegend(handles, labels, order):
  assert len(handles) == len(labels) and len(labels) == len(order)
  reordered_handles = []
  reordered_labels = []
  for i in order:
    reordered_handles.append(handles[i])
    reordered_labels.append(labels[i])
  return reordered_handles, reordered_labels


def filterValues(x, y, x_interval):
  filter = (x >= x_interval[0]) * (x <= x_interval[1])
  return x[filter], y[filter]


def plotCDF(list, plt, style, label, bins=100):
  count, edges = np.histogram(list, bins=bins, density=True)
  cdf = np.cumsum(count)
  plt.plot(edges[1:], cdf / cdf[-1], style, label=label)
  plt.set_ylim(0, 1)


def choose(n, k):
  if k > n // 2: k = n - k
  p = Fraction(1)
  for i in range(1, k + 1):
    p *= Fraction(n - i + 1, i)
  return int(p)


def isSliceFullBuffer(s, ue_conf):
  '''
  Checks if Slice s has UEs with full_buffer
  :param s: slice id
  :param ue_conf: UE configurations
  :return: True if at least one UE configuration belonging to that slice is full_buffer
  '''
  for u in ue_conf:
    if u['traffic'] == 'full_buffer' and u['slice'] == s:
      return True
  return False

def getSINRfromDistance(d2d, los=True, env="macro", config_params=None):
  params = {
    'h_ue': 1.5,  # m
    'h_enb': 20,  # m
  }
  if config_params is not None:
    params.update(config_params)
  enb = Medium.ENB(params['h_enb'])
  ue = Medium.UE(params['h_ue'])
  snr_calculator = Medium.SNRCalculator(None, enb, ue)
  lobes = [0, 120, 240]
  sinr_array = []
  if d2d == 0:
    d2d = 0.00001
  # check all antenna lobes and return the maximum
  for lobe in lobes:
    sinr_array.append(snr_calculator.getSNR((0, 0, params['h_enb']), (0, d2d, params['h_ue']), lobe, pre_los=los))
  return max(sinr_array)


def slice_metric_aggregator(values, fun=np.mean):
  keys = set()
  for v in values:
    for k in v:
      keys.add(k)
  return dict(map((lambda s: (s, fun(np.fromiter(map((lambda x: x[s] if s in x else 0), values), dtype="float")))), keys))


def check_reliability(unreliable, total, target):
  if unreliable == 0 or (unreliable == 1 and target != 0):
    return True
  elif unreliable/total > target:
    return False


class CircularIterator:

  def __init__(self, lst, root=0):
    self.lst = lst
    self.root = root

  def __iter__(self):
    self.current = self.root
    self.first = True
    return self

  def __next__(self):
    if len(self.lst) == 0:
      raise StopIteration
    curr = self.current
    self.current = (self.current + 1) % len(self.lst)
    if curr == self.root and not self.first:
      raise StopIteration
    else:
      self.first = False
      return self.lst[curr]


class BucketAdmission:
  def __init__(self, rate, capacity, tti):
    self.rate = rate * 1000  # rate in kbytes per second
    self.capacity = capacity
    self.tti = tti
    self.tokens = capacity
    self.ts = 1

  def tick(self, ticks=1):
    tokens_to_add = int(self.rate * self.tti)
    self.tokens = min(self.tokens + tokens_to_add * ticks, self.capacity)

  def filter_packet(self, size):
    if self.tokens - size > 0:
      self.tokens -= size
      accepted = True
    else:
      accepted = False
    return accepted


class FineArray:
  def __init__(self, x_interval=100, aggr_fun=np.mean, error_fun=np.std, continuous=True,
               empty_value=np.nan, empty_error=np.nan):
    self.fine_values = []
    self.this_x = None
    self.last_x = None
    self.values = []
    self.err = []
    self.x = []
    self.aggr_fun = aggr_fun
    self.error_fun = error_fun
    self.x_interval = x_interval
    self.continuous = continuous
    self.empty_value = empty_value
    self.empty_error = empty_error

  def __getstate__(self):
    obj = self.__dict__
    obj['aggr_fun'] = None
    obj['error_fun'] = None
    return obj

  def fine_append(self, x, y):
    self.this_x = (x // self.x_interval) * self.x_interval if self.this_x is None else self.this_x
    aggregated_value = (None, None, None)
    new_coarse_x = (x // self.x_interval) * self.x_interval
    assert x >= self.this_x
    if new_coarse_x != self.this_x:
      aggregated_value = self.aggregate(x)
      self.this_x = new_coarse_x
    self.last_x = x
    self.fine_values.append(y)
    return aggregated_value

  def aggregate(self, new_x=None):
    if self.fine_values:
      self.x.append(self.this_x)
      value = self.aggr_fun(self.fine_values)
      self.values.append(value)
      if self.error_fun is not None:
        err = self.error_fun(self.fine_values)
        self.err.append(err)
      else:
        err = None
      self.fine_values.clear()
      if new_x is not None:
        skips = (new_x - self.this_x) // self.x_interval - 1
        if skips > 0:
          for n in range(0, skips):
            coarse_x = (self.this_x // self.x_interval + (n + 1)) * self.x_interval
            self.x.append(coarse_x)
            self.values.append(self.empty_value)
            self.err.append(self.empty_error)
      return self.this_x, value, err
    else:
      return None, None, None

  def append(self, y):
    if self.last_x is None:
      x = 0
    else:
      x = self.last_x + 1
    self.fine_append(x, y)

  def get_values(self):
    return self.values

  def get_fine_array(self):
    return self.fine_values

  def get_x(self):
    return self.x

  def get_error(self):
    return self.err

  def get_x_and_y(self, final=True):
    if final:
      self.aggregate()
    return {
      'x': self.x,
      'y': self.values
    }

  def __getitem__(self, item):
    return self.values[item]

  def __len__(self):
    return len(self.values)

  def get_last_values(self, x_gap):
    last_n = int(x_gap/self.x_interval)
    return self.values[-last_n:]


class PandasStreamer:

  def __init__(self, csv_file, chunk_size=100, offset=0, search_boost=10000, value_filter=None, stored_columns=None):
    self.file = csv_file
    self.chunk_size = chunk_size
    self.last_offset = offset
    self.offset = offset
    self.df = None
    self.columns = None
    self.search_boost = search_boost
    self.filter = value_filter
    self.stored_columns = stored_columns
    self.columns = None
    self.chunk_index = None
    self.interruption = False
    self.empty = False
    self._load_chunk()

  def _load_chunk(self, return_boost=False, use_filter_boost=True):
    if return_boost or self.filter and use_filter_boost:
      chunk = self.search_boost
    else:
      chunk = self.chunk_size
    self.df = pd.read_csv(self.file, skiprows=self.offset, nrows=chunk, names=self.columns)
    self.chunk_index = self.df.index
    lines_read = len(self.df)
    if self.columns is None:
      lines_read += 1
      self.columns = self.df.columns
    if lines_read > 0:
      last_index = self.df.index[-1]
      if self.filter:
        past_lines_read = 0
        while True:
          for k in self.filter:
            filter_value = self.filter[k]
            if type(filter_value) is list:
              condition = self.df[k].isin(filter_value)
            else:
              condition = self.df[k] == filter_value
            self.df = self.df[condition]
          if len(self.df) >= chunk:
            self.df = self.df.iloc[:chunk]
            idx = self.df.index[chunk - 1]
            lines_read = past_lines_read + idx
            break
          else:
            past_lines_read = lines_read
            new_df = pd.read_csv(self.file, skiprows=lines_read, nrows=chunk, names=self.columns)
            lines_read += len(new_df)
            if new_df.empty:
              break
            new_df.index += last_index + 1
            last_index = new_df.index[-1]
            self.df = self.df.append(new_df)
        self.chunk_index = self.df.index
        self.df = self.df.reset_index()
    if self.stored_columns:
      self.df = self.df[self.stored_columns]
    self.last_offset = self.offset
    if self.last_offset == 0:
      self.last_offset += 1  # header line
    self.offset += lines_read
    self.empty = (len(self.df) == 0)

  def get_next(self):
    if self.df.empty and not self.empty:
      self._load_chunk()
    return self.df.iloc[0] if not self.df.empty else None

  def pop(self):
    if self.df.empty and not self.empty:
      self._load_chunk()
    nxt = self.get_next()
    if not self.df.empty:
      self.df = self.df[1:]
    return nxt

  def sorted_skip_until(self, column, value, use_boost=True, return_columns=None):
    chunks_loaded = False
    if return_columns is not None:
      output_entries = pd.DataFrame(columns=return_columns)
    else:
      output_entries = None
    while not self.df.empty and self.df.iloc[-1][column] < value:
      chunks_loaded = True
      if return_columns is not None:
        output_entries = output_entries.append(self.df[return_columns], ignore_index=True)
      self._load_chunk(return_boost=use_boost)
    if not self.df.empty:
      idx = self.df[column].searchsorted(value, 'left')
      df_value = self.df.iloc[idx][column]
      # ideally it would be searchsorted(x,'right') - 1, but only 'left' allows us to see
      #  if more chunks are required, this fixes the discrepancy
      if df_value != value and idx > 0:
        idx -= 1
      if use_boost and chunks_loaded:
        self.offset = self.last_offset + self.chunk_index[idx]
        if return_columns is not None:
          output_entries = output_entries.append(self.df.iloc[0:idx][return_columns], ignore_index=True)
        self._load_chunk()
      else:
        self.df = self.df.iloc[idx:]
        if return_columns is not None:
          output_entries = output_entries.append(self.df.iloc[0:idx][return_columns], ignore_index=True)
    return output_entries

  def slip_next(self, column, value, use_boost=True):
    self.df = self.df[self.df[column] != value]
    while self.df.empty:
      self._load_chunk(return_boost=use_boost)
      if self.df.empty:
        break
      self.df = self.df[self.df[column] != value]
    # Load again even if boost is not used, since we want to undo the filter. This is not the best in terms of
    # performance but its usage is not expected to be used a lot. Currently, only in handovers to non-existing cells.
    if not self.df.empty:
      idx = self.df.index[0]
      self.offset = self.last_offset + self.chunk_index[idx]  # +1 is header line
      self._load_chunk()

  def is_empty(self):
    if self.df.empty:
      nxt = self.get_next()
      return nxt is None
    else:
      return False

  # get last entry for begining and ending porpuses, does not affect the stream and offset count
  def get_last_entry(self):
    df = pd.read_csv(self.file, skiprows=self.last_offset, names=self.columns)
    if df.empty:
      return None
    else:
      return df.iloc[-1]


class DebuggerStopwatch:

  def __init__(self):
    self.stopwatches = {}

  def add_class(self, class_name, aggregation_function=sum):
    self.stopwatches[class_name] = {
      'last': None,
      'values': [],
      'aggregation': aggregation_function
    }

  def tick(self, class_name=None, continuous=False):
    stamp = time.time()
    if class_name not in self.stopwatches:
      self.add_class(class_name)
    classes = [class_name] if class_name is not None else list(self.stopwatches.keys())
    for c in classes:
      sw = self.stopwatches[c]
      if sw['last'] is not None:
        sw['values'].append(stamp - sw['last'])
        if not continuous:
          sw['last'] = None
        else:
          sw['last'] = stamp
      else:
        sw['last'] = stamp

  def clear(self, class_name=None):
    classes = [class_name] if class_name is not None else list(self.stopwatches.keys())
    for c in classes:
      sw = self.stopwatches[c]
      sw['last'] = None
      sw['values'].clear()

  def show(self, class_name=None):
    classes = [class_name] if class_name is not None else list(self.stopwatches.keys())
    if len(classes) == 1:
      output = None
    else:
      output = {}
    for c in classes:
      sw = self.stopwatches[c]
      final_value = sw['aggregation'](sw['values'])
      if output is None:
        output = final_value
      else:
        output[c] = final_value
    return output

  def show_and_clear(self, class_name=None):
    output = self.show(class_name)
    self.clear()
    return output

