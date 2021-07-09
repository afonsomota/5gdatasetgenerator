# Dataset generator for wireless mobile networks


Currently this is a placeholder for the code of the dataset generator and the example dataset generator. Files are currently being sorted and organized for publishing.

If the paper is accepted, the code will be uploaded to Zenodo or similar platform so that it receives a DOI.

## Using OSM Web Wizard

Usage of the OSM Web Wizard can be looked up in the [SUMO website](https://sumo.dlr.de/docs/Tutorials/OSMWebWizard.html). Summing up, inside the SUMO directory there is a folder `tools`, which contains a file named `osmWebWizard.py`. By running it using Pyhton, a web interface pops up with the OSM Web Wizard. From it, a SUMO scenario can be downloaded. For the scope of this work, it is important to export building information as well with the tick in `Add polygons`. The file `osm.sumocfg` will be the input to the SUMO simulator. The file `osm.poly` will be the input to the "Map processing" (`Medium.py`).

## Using SUMO

For using sumo, the following comand should be used inside the folder exported by the OSM Web Wizard, where `<granularity>` is the granularity of the output. We used `0.1`. A file named `fcd.xml` will be used as the input to the "Node processing" (`PowerTracker.py`).

```
sumo -c osm.sumocfg --fcd-output fcd.xml --device.fcd.period <granularity> --step-length <granularity>
```
