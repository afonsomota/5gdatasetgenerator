from Generators import Web, OfflineVideo, VoIP, MTCSensors, V2I, V2IBurst

traffic_classes = {
  "web": {
    "class": Web,
    "config": [
      {
        "probability": 1,
        "value": {
          "inter_session": 600,
          "pages_per_session": 3.86,
          "pages_per_session_std": 2.465
        }
      }
    ]
  },
  "web-sporadic": {
    "class": Web,
    "config": [
      {
        "probability": 1,
        "value": {
          "inter_session": 1200,
          "pages_per_session": 3.86,
          "pages_per_session_std": 2.465
        }
      }
    ]
  },
  "web-constant": {
    "class": Web,
    "config": [
      {
        "probability": 1,
        "value": {
          "inter_session": 10,
          "pages_per_session": 3.86,
          "pages_per_session_std": 2.465
        }
      }
    ]
  },
  "offline_video": {
    "class": OfflineVideo,
    "config": [
      {
        "probability": 1,
        "value": {
          "inter_video_session": 1200,
          "video_session_duration": 1200
        }
      }
    ]
  },
  "offline_video-constant": {
    "class": OfflineVideo,
    "config": [
      {
        "probability": 1,
        "value": {
          "inter_video_session": 0,
          "video_session_duration": 1200
        }
      }
    ]
  },
  "offline_video-sporadic": {
    "class": OfflineVideo,
    "config": [
      {
        "probability": 1,
        "value": {
          "inter_video_session": 2400,
          "video_session_duration": 600
        }
      }
    ]
  },
  "voip": {
    "class": VoIP,
    "config": [
      {
        "probability": 1,
        "value": {
          "inter_call": 600,
          "average_call_duration": 180
        }
      }
    ]
  },
  "voip-constant": {
    "class": VoIP,
    "config": [
      {
        "probability": 1,
        "value": {
          "inter_call": 10,
          "average_call_duration": 180
        }
      }
    ]
  },
  "voip-sporadic": {
    "class": VoIP,
    "config": [
      {
        "probability": 1,
        "value": {
          "inter_call": 1200,
          "average_call_duration": 60
        }
      }
    ]
  },
  "mtc": {
    "class": MTCSensors,
    "config": [
      {
        "probability": 1,
        "value": {
          "inter_packet_time": 234,
          "packet_size": 200
        }
      }
    ]
  },
  "v2i": {
    "class": V2I,
    "config": [
      {
        "probability": 1,
        "value": {}
      }
    ]
  },
  "v2i_burst": {
    "class": V2IBurst,
    "config": [
      {
        "probability": 1,
        "value": {
          "burst_size": 320
        }
      }
    ]
  }
}

scenarios = {
  "standard": {
    'static': [
      {
        'probability': 0.4*0.6,
        'model': ["web"],
        'slice': 2
      },
      {
        'probability': 0.2*0.6,
        'model': ["voip"],
        'slice': 4
      },
      {
        'probability': 0.4,
        'model': ["mtc"],
        'slice': 3
      },
      {
        'probability':  0.2*0.6,
        'model': ["offline_video"],
        'slice': 2
      },
      {
        'probability':  0.2*0.6,
        'model': [],
        'slice': -1
      },
    ],
    'vehicle': [
      {
        'probability': 1,
        'model': ["v2i", "v2i_burst"],
        'slice': 1
      }
    ],
    'passenger': [
      {
        'probability': 0.4,
        'model': ["web"],
        'slice': 2
      },
      {
        'probability': 0.2,
        'model': ["voip"],
        'slice': 4
      },
      {
        'probability': 0.2,
        'model': [],
        'slice': -1
      },
      {
        'probability': 0.2,
        'model': ["offline_video"],
        'slice': 2
      }
    ],
    'pedestrian': [
      {
        'probability': 0.4,
        'model': ["web"],
        'slice': 2
      },
      {
        'probability': 0.2,
        'model': ["voip"],
        'slice': 4
      },
      {
        'probability': 0.2,
        'model': [],
        'slice': -1
      },
      {
        'probability': 0.2,
        'model': ["offline_video"],
        'slice': 2
      }
    ]
  },
  "undistributed": {
    'static': [
      {
        'probability': 0.25*0.4*0.6,
        'model': ["web-sporadic"],
        'slice': 2
      },
      {
        'probability': 0.5*0.4*0.6,
        'model': ["web-constant"],
        'slice': 2
      },
      {
        'probability': 0.25*0.2*0.6,
        'model': ["voip-sporadic"],
        'slice': 4
      },
      {
        'probability': 0.5*0.2*0.6,
        'model': ["voip-constant"],
        'slice': 4
      },
      {
        'probability':  0.25*0.2*0.6,
        'model': ["offline_video-sporadic"],
        'slice': 2
      },
      {
        'probability': 0.5*0.2*0.6,
        'model': ["offline_video-constant"],
        'slice': 2
      },
      {
        'probability':  (0.2 + 0.25*0.8)*0.6,
        'model': [],
        'slice': -1
      },
      {
        'probability': 0.4,
        'model': ["mtc"],
        'slice': 3
      },
    ],
    'vehicle': [
      {
        'probability': 1,
        'model': ["v2i", "v2i_burst"],
        'slice': 1
      }
    ],
    'passenger': [
      {
        'probability': 0.25 * 0.4,
        'model': ["web-sporadic"],
        'slice': 2
      },
      {
        'probability': 0.5 * 0.4,
        'model': ["web-constant"],
        'slice': 2
      },
      {
        'probability': 0.25 * 0.2,
        'model': ["voip-sporadic"],
        'slice': 4
      },
      {
        'probability': 0.5 * 0.2,
        'model': ["voip-constant"],
        'slice': 4
      },
      {
        'probability': 0.25 * 0.2,
        'model': ["offline_video-sporadic"],
        'slice': 2
      },
      {
        'probability': 0.5 * 0.2,
        'model': ["offline_video-constant"],
        'slice': 2
      },
      {
        'probability': 0.2 + 0.25 * 0.8,
        'model': [],
        'slice': -1
      },
    ],
    'pedestrian': [
      {
        'probability': 0.25 * 0.4,
        'model': ["web-sporadic"],
        'slice': 2
      },
      {
        'probability': 0.5 * 0.4,
        'model': ["web-constant"],
        'slice': 2
      },
      {
        'probability': 0.25 * 0.2,
        'model': ["voip-sporadic"],
        'slice': 4
      },
      {
        'probability': 0.5 * 0.2,
        'model': ["voip-constant"],
        'slice': 4
      },
      {
        'probability': 0.25 * 0.2,
        'model': ["offline_video-sporadic"],
        'slice': 2
      },
      {
        'probability': 0.5 * 0.2,
        'model': ["offline_video-constant"],
        'slice': 2
      },
      {
        'probability': 0.2 + 0.25 * 0.8,
        'model': [],
        'slice': -1
      },
    ],
  },
  "sporadic": {
    'static': [
      {
        'probability': 0.4 * 0.6,
        'model': ["web-sporadic"],
        'slice': 2
      },
      {
        'probability': 0.2 * 0.6,
        'model': ["voip-sporadic"],
        'slice': 4
      },
      {
        'probability': 0.4,
        'model': ["mtc"],
        'slice': 3
      },
      {
        'probability': 0.2 * 0.6,
        'model': ["offline_video-sporadic"],
        'slice': 2
      },
      {
        'probability': 0.2 * 0.6,
        'model': [],
        'slice': -1
      },
    ],
    'vehicle': [
      {
        'probability': 1,
        'model': ["v2i", "v2i_burst"],
        'slice': 1
      }
    ],
    'passenger': [
      {
        'probability': 0.4,
        'model': ["web-sporadic"],
        'slice': 2
      },
      {
        'probability': 0.2,
        'model': ["voip-sporadic"],
        'slice': 4
      },
      {
        'probability': 0.2,
        'model': [],
        'slice': -1
      },
      {
        'probability': 0.2,
        'model': ["offline_video-sporadic"],
        'slice': 2
      }
    ],
    'pedestrian': [
      {
        'probability': 0.4,
        'model': ["web-sporadic"],
        'slice': 2
      },
      {
        'probability': 0.2,
        'model': ["voip-sporadic"],
        'slice': 4
      },
      {
        'probability': 0.2,
        'model': [],
        'slice': -1
      },
      {
        'probability': 0.2,
        'model': ["offline_video-sporadic"],
        'slice': 2
      }
    ]
  }
}


def get_default_configuration(mode="standard", out_folder="traffic-traces-standard", static=0.005, timeout=30400):
  return {
    "static": {
      "density": static,  # per m^2
      "inside": 0.5,  # % inside
      "poi": []  # {'probability': 0.1, 'location': [123,232], 'radius', 20}
    },
    "overwrite": False,
    "timeout": timeout,
    "trace_folder": "small-berlin-long-out/",
    "scene_conf_file": "small-berlin-long-out/sim.config",
    "snr_file": "small-berlin-long-out/max_snr_per_pos.pkl",
    "output_folder": out_folder,
    "sampling_period": 0.1,
    "snr_error_std": 0.1,
    "bursts": {
      "inter_burst": 2,
      "radius": 100
    },
    "car_passenger_generator": None,
    "traffic": traffic_classes,
    'attribution': scenarios[mode]
  }
