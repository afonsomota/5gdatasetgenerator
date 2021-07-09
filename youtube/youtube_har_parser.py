import sys
import os
import json
from pathlib import Path
import datetime
import glob
import numpy as np
import scipy
import scipy.stats


if __name__ == '__main__':

  names_dir = '<HAR_FILES_DIR>'
  out_dir = 'youtube_traces'

  if len(sys.argv) > 1:
    names = []
    for i in range(1,len(sys.argv)):
      names.append(sys.argv[1])
  else:
    names = glob.glob(names_dir+"*.har")

  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  for name in names:
    try:
      f = open(name)
      data = json.load(f)
    except:
      print("Error parsing",name)
      continue
    out_file = open(str(Path(out_dir, Path(name).stem + ".csv")), "w")
    conf_file = open(str(Path(out_dir, Path(name).stem + ".cnf")), "w")
    conf_obj = {
      'inter_packet_distribution': {},
      'packets_sizes': []
    }
    start_dt = None
    inter_packet_list = []
    last_dt = None
    video_len = None
    for ev in data['log']['entries']:
      if 'url' in ev['request'] and '/videoplayback' in ev['request']['url']:
        ev_dt = datetime.datetime.strptime(ev['startedDateTime'], "%Y-%m-%dT%H:%M:%S.%fZ")
        if start_dt is None:
          start_dt = ev_dt
          seconds_lapsed = 0
        else:
          seconds_lapsed = (ev_dt - start_dt).total_seconds()
          inter_packet = (ev_dt - last_dt).total_seconds()
          inter_packet_list.append(inter_packet)
        conf_obj['packets_sizes'].append(int(ev['response']['_transferSize']))
        last_dt = ev_dt
        print(seconds_lapsed, ev['response']['_transferSize'], sep=",", file=out_file)
      elif video_len is None and 'url' in ev['request'] and '/watchtime' in ev['request']['url']:
        for param in ev['request']['queryString']:
          if param['name'] == 'len':
            video_len = float(param['value'])

    inter_packet_list = np.array(inter_packet_list)
    dist = scipy.stats.lognorm
    param = dist.fit(inter_packet_list)
    conf_obj['inter_packet_distribution']['distribution'] = "lognorm"
    conf_obj['inter_packet_distribution']['params'] = param
    conf_obj['video_length'] = video_len
    json.dump(conf_obj, conf_file)

