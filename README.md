# Dataset generator for wireless mobile networks


Currently this is a placeholder for the code of the dataset generator and the example dataset generator. Files are currently being sorted and organized for publishing. Still exploring options for the generated dataset upload. The SNR traces, which have 1ms granularity occupy more than 150GB.

If the paper is accepted, the code will be uploaded to Zenodo or similar platform so that it receives a DOI.

## Using OSM Web Wizard

Usage of the OSM Web Wizard can be looked up in the [SUMO website](https://sumo.dlr.de/docs/Tutorials/OSMWebWizard.html). Summing up, inside the SUMO directory there is a folder `tools`, which contains a file named `osmWebWizard.py`. By running it using Pyhton, a web interface pops up with the OSM Web Wizard. From it, a SUMO scenario can be downloaded. For the scope of this work, it is important to export building information as well with the tick in `Add polygons`. The file `osm.sumocfg` will be the input to the SUMO simulator. The file `osm.poly` will be the input to the "Map processing" (`Medium.py`).

## Using SUMO

For using sumo, the following comand should be used inside the folder exported by the OSM Web Wizard, where `<granularity>` is the granularity of the output. We used `0.1`. A file named `fcd.xml` will be used as the input to the "Node processing" (`PowerTracker.py`).

```
sumo -c osm.sumocfg --fcd-output fcd.xml --device.fcd.period <granularity> --step-length <granularity>
```

## Map processing (`Medium.py`)

The usage for the Map processing module is as shown below. The `<polygon_file>` parameter is the path to the `osm.poly` within the scenario from OSM Web Wizard. `<center_x>` and `<center_y>` are the coordinates where the central base station will be generated. `<working_folder>` is where the final static maps (one global and one for each cell) will be placed. 

```
python Medium.py <polygon_file> <center_x> <center_y> <working_folder>
```

For the generated trace, the values of 1080.0, 892.0 were used for `<center_x>`, `<center_y>`.

## Node processing (`PowerTracker.py`)

The usage for the Node processing module is show below. `<fcd_file>` is the `fcd.xml` file generated by sumo. `<working_folder>` is the same as used by the Map processing to export the maps. `<t_start>` is the timestamp at which the node processing should start. `<t_stop>` is the timestamp at which the node processing should stop.

```
python PowerTracker.py <fcd_file> <working_folder> <t_start> <t_stop>
```

## Network traffic generation (`Generator.py`)

The input configurations for the traffic generator (including the node trace folder, i.e. the working direction, are inside `Configurations.py`.
