import sys
import pickle
import math
import numpy as np
from scipy.spatial import ConvexHull
import scipy
import xml.etree.ElementTree as ET
import os

from Util_map import Util

channel_params = {
  'UMa_A': {
    'K-factor': {
      'miu_k': 9,
      'sigma_k': 3.5
    },
    'f_c': 4*10**9,
    'h_ue': 1.5,
    'h_bs': 25
  }
}

class Node:
  def __init__(self, height, nf, gain, txpower):
    self.height = height
    self.nf = nf  # noise factor
    self.gain = gain  # antenna isotropic gain
    self.txpower = txpower

  def directional_gain(self, tx_pos, rx_pos, cell_lobe):
    return self.gain


class UE(Node):
  def __init__(self, hue=1.5):
    super().__init__(hue, 7, 0, 23)


class MPFadingModel:
  def __init__(self, t_sim, f_max, f_c, sigma_0, rho, f_rho=0, theta_rho=0, n_i=200, t_s=0.001, t_start=0):
    self.n_i = n_i
    self.t_s = t_s
    self.t_sim = t_sim
    self.f_max = f_max
    self.f_c = f_c
    self.sigma_0 = sigma_0
    self.rho = rho
    self.f_rho = f_rho
    self.theta_rho=theta_rho
    self.t_start = t_start

  def run(self):
    no_samples = np.ceil(self.t_sim / self.t_s)
    t = np.arange(0, no_samples) * self.t_s

    if self.f_max == 0:
      xi_t = np.ones(t.shape)
      mu_i = np.ones(t.shape)
    else:
      c_1, f_1, theta_1 = MPFadingModel.params_meds(self.n_i, self.f_max, self.sigma_0)
      c_2, f_2, theta_2 = MPFadingModel.params_meds(self.n_i + 1, self.f_max, self.sigma_0)

      mu_i = 0
      mu_q = 0

      for n in range(self.n_i):
        mu_i += c_1[n] * np.cos(2*np.pi*f_1[n]*t+theta_1[n])
        mu_q += c_2[n] * np.cos(2*np.pi*f_2[n]*t+theta_2[n])

      m_i = self.rho * np.cos(2*np.pi*self.f_rho*t+self.theta_rho)
      m_q = self.rho * np.sin(2*np.pi*self.f_rho*t+self.theta_rho)
      xi_t = np.abs(mu_i + m_i + 1j * (mu_q + m_q))

    return t + self.t_start, xi_t, mu_i

  @staticmethod
  def params_meds(n_i, f_max, sigma_0):
    n = np.arange(0, n_i).T
    c_i_n = sigma_0 * np.sqrt(2/n_i) * np.ones(len(n))
    f_i_n = f_max * np.sin(np.pi / (2 * n_i) * (n - 1/2))
    theta_i_n = 2 * np.pi * np.random.rand(n_i, 1)
    if f_max == 0:
      c_i_n = 0
    return c_i_n, f_i_n, theta_i_n




class ENB(Node):

  def __init__(self, hbs=25):
    self.angle3db = 65
    self.maximum_attenuation = 30
    super().__init__(hbs, 5, 8, 46)

  def horizontal_gain(self, angle, ma):
    """Calculate horizontal gain based in horizontal angle and maximum attenuation (ma)"""
    return -min(12 * (angle / self.angle3db) ** 2, ma)

  def vertical_gain(self, angle, ma):
    """Calculate vertical gain based in vertical angle and maximum attenuation (ma)"""
    return -min(12 * ((angle - 90) / self.angle3db) ** 2, ma)

  def directional_gain(self, tx_pos, rx_pos, cell_lobe):
    rp = (rx_pos[0] - tx_pos[0], rx_pos[1] - tx_pos[1], rx_pos[2] - tx_pos[2])
    clr = math.radians(cell_lobe)
    d = math.sqrt(rp[0] ** 2 + rp[1] ** 2 + rp[2] ** 2)
    hangle = math.degrees(math.atan2(rp[0], rp[1])) - cell_lobe
    if hangle > 180: hangle -= 360
    if hangle < -180: hangle += 360
    vangle = math.degrees(math.asin(rp[2] / d)) + 90
    hgain = self.horizontal_gain(hangle, self.maximum_attenuation)
    vgain = self.vertical_gain(vangle, self.maximum_attenuation)
    return -min(-abs(vgain + hgain), self.maximum_attenuation)


# hangle = math.radians(math.atan2(rp[0],rp[1])) - cell_lobe

class Medium:
  def __init__(self):
    self.carrier = 4
    self.tn = 174  # thermal noise
    self.sw = 20  # street width


class Building:
  def __init__(self, vertex, height):
    self.importAsConvexHull(vertex)
    self.height = height
    self.faceNormals = []
    self.faceVertex = []
    self.calculateFacesNormals()
    self.calculateFacesVertex()

  def importAsConvexHull(self, vertex):
    vertex2d = list(map(lambda x: [x[0], x[1]], vertex))
    hull = ConvexHull(vertex2d)
    cw_vertices = list(hull.vertices)
    cw_vertices.reverse()
    self.vertex = list(map(lambda i: [hull.points[i][0], hull.points[i][1], 0], cw_vertices))
    return

  def calculateFacesNormals(self):
    normals = []
    for i in range(0, len(self.vertex)):
      ni = i + 1 if i != len(self.vertex) - 1 else 0
      vi0 = np.array(self.vertex[i])
      vih = np.array([self.vertex[i][0], self.vertex[i][1], self.height])  # vertex top
      vni0 = np.array(self.vertex[ni])
      nrm = np.cross(vih - vi0, vni0 - vi0)
      if not np.array_equal(nrm, np.array([0, 0, 0])):
        nrm = nrm / np.linalg.norm(nrm)
      normals.append(nrm)
    normals.append(np.array([0, 0, 1]))  # normal to roof
    normals.append(np.array([0, 0, -1]))  # normal to roof
    self.faceNormals = normals
    return

  def calculateFacesVertex(self):
    fvertex = list(map(np.array, self.vertex))
    fvertex.append(np.array([self.vertex[0][0], self.vertex[0][1], self.height]))  # add one roof vertice
    fvertex.append(np.array(self.vertex[0]))
    self.faceVertex = fvertex
    return

  def getFacesNormals(self):
    return self.faceNormals

  def getFacesVertex(self):
    return self.faceVertex

  def calculateIntersection(self, tx_pos, rx_pos):
    """Intersection of segment with polyhedron. Source: http://geomalgorithms.com/a13-_intersect-4.html"""
    p0 = np.array(tx_pos)
    p1 = np.array(rx_pos)

    if np.array_equal(p1, p0): return False, False, 0

    te = 0  # max entering
    tl = 1  # max leaving
    ds = p1 - p0
    n = self.getFacesNormals()
    v = self.getFacesVertex()
    leaving = False
    entering = False

    # for each face
    for i in range(0, len(v)):
      N = - np.dot(p0 - v[i], n[i])
      D = np.dot(ds, n[i])

      # line parallel to face
      if D == 0:
        # if n is 0, segment will never touch polyhedron
        if N < 0:
          return False, False, 0
        continue

      t = N / D
      if D < 0:
        if t > te:
          entering = True
          te = t
          if te > tl:
            return False, False, 0

      if D > 0:
        if t < tl:
          leaving = True
          tl = t
          if tl < te:
            return False, False, 0

    assert te < np.linalg.norm(ds)
    # if tl > np.linalg.norm(ds):
    #   return False
    if leaving and entering:  # both ends are outside of the building
      return True, False, 0
    else:
      if entering:
        pe = p0 + te*ds
        d_2d_in = np.linalg.norm(pe[0:2]-p1[0:2])
      else:
        pl = p0 + tl * ds
        d_2d_in = np.linalg.norm(pl[0:2] - p0[0:2])
      return True, True, d_2d_in


class SNRCalculator:
  def __init__(self, simulation, bmap=None, bs=None, ue=None, building_height=15):
    self.ue = ue if ue is not None else UE()
    self.enb = bs if bs is not None else ENB()
    self.simulation = simulation
    self.medium = Medium()
    self.building_height = building_height
    self.obstacles = []
    if bmap is not None:
      bmapf = bmap[0]
      bmapc = bmap[1]
      self.loadMap(bmapf, bmapc)

  def addObstacle(self, obs):
    self.obstacles.append(obs)

  def clearObstacles(self):
    self.obstacles.clear()

  # 2500,1250
  def loadMap(self, bmap, center):
    """Load building map into an obstacle list"""
    limits = self.simulation.get_map_limits()
    tree = ET.parse(bmap)
    root = tree.getroot()
    polys = root.findall('.//poly')
    for p in polys:
      t = p.get('type')
      if 'building' in t:
        vertex = list(map(lambda x: [float(x.split(",")[0]) - center[0], float(x.split(",")[1]) - center[1], 0],
                          p.get('shape').split(" ")))
        np_vtx = np.array(vertex)
        is_inside = np.array([
          np_vtx[:, 0] >= limits[0][0],  # x min
          np_vtx[:, 0] <= limits[0][1],  # x max
          np_vtx[:, 1] >= limits[1][0],  # y min
          np_vtx[:, 1] <= limits[1][1],  # y max
        ]).all(axis=0).any()  # at least one vertex is within the simulation area
        if is_inside and len(vertex) > 2:
          obs = Building(vertex, self.building_height)
          self.addObstacle(obs)
    return

  def checkLOS(self, tx_pos, rx_pos):
    """Check obstacles in the way of the transmission, return 1 for LOS and 0 for NLOS"""
    los = True
    building_height = 0
    is_inside = False
    d_2d_in = 0
    for obs in self.obstacles:
      obj_int, obj_inside, d_2d_in = obs.calculateIntersection(tx_pos, rx_pos)
      if obj_int:
        los = False
        building_height = max(obs.height, building_height)
        is_inside = is_inside or obj_inside
        if d_2d_in > 0:
          break
        #return False, obs.height,
    return los, building_height, is_inside, d_2d_in

  def getPathLossUMa(self, tx_pos, rx_pos, pre_los=None, pre_h=0):
    """Get the Path Loss between two nodes. Returns pathloss and shadow fading std-dev"""
    if pre_los is None:
      (los, h, inside, d_2d_in) = self.checkLOS(tx_pos, rx_pos)
    else:
      los = pre_los[0]
      inside = pre_los[1]
      d_2d_in = pre_los[2]
      h = pre_h
    d3 = math.sqrt((tx_pos[0] - rx_pos[0]) ** 2 +
                   (tx_pos[1] - rx_pos[1]) ** 2 +
                   (tx_pos[1] - rx_pos[1]) ** 2
                   )
    d2 = math.sqrt((tx_pos[0] - rx_pos[0]) ** 2 +
                   (tx_pos[1] - rx_pos[1]) ** 2
                   )
    htx = tx_pos[2]
    hrx = rx_pos[2]
    if htx > hrx:
      hbs = htx
      hue = hrx
    else:
      hbs = hrx
      hue = htx
    dbp = 4 * (htx - 1) * (hrx - 1) * self.medium.carrier * 10 / 3.

    if d2 < dbp:
      pl_los = 28 + 22 * math.log10(d3) + 20 * math.log10(self.medium.carrier)
    elif d2 >= dbp:
      pl_los = 40 * math.log10(d3) + 28 + 20 * math.log10(self.medium.carrier) - \
               9 * math.log10((dbp) ** 2 + (htx - hrx) ** 2)

    if los:
      pl = pl_los
      sf = 4
      h = 0
    else:
      assert h > 0
      pl_nlos = 161.04 - 7 * math.log10(self.medium.sw) + 7.5 * math.log10(h) - \
                (24.37 - 3.7 * (h / hbs) ** 2) * math.log10(hbs) + \
                (43.42 - 3.1 * math.log10(hbs)) * (math.log10(d3) - 3) + \
                20 * math.log10(self.medium.carrier) - \
                (3.2 * (math.log10(17.625)) ** 2 - 4.97) - \
                0.6 * (hue - 1.5)
      pl_nlos = 13.54 + 39.08 * math.log10(d3) + 20 * math.log10(self.medium.carrier) - 0.6 * (hue - 1.5)
      pl = max(pl_los, pl_nlos)
      sf = 6
    if inside:
      pl += 20 + 0.5*d_2d_in
      sf = 7

    return pl, sf, (los, inside, d_2d_in), h

  def getSNR(self, tx_pos, rx_pos, cell_lobe, direction="downlink", pre_los=None, pre_h=20):
    """Get the SINR in a direction  between two nodes according to their position"""
    if direction == "downlink":
      src = self.enb
      dst = self.ue
    else:
      src = self.ue
      dst = self.enb

    pl = self.getPathLossUMa(tx_pos, rx_pos, pre_los, pre_h)

    pr = src.txpower + src.directional_gain(tx_pos, rx_pos, cell_lobe) + dst.directional_gain(rx_pos, tx_pos,
                                                                                              cell_lobe) - pl[0]

    snr = pr - dst.nf + self.medium.tn - 10 * math.log10(20 * 1000000)

    return (snr,
            pl[1],# + np.random.normal(0, pl[2]),
            pl[2],
            pl[3]
            )


class Simulation:

  def __init__(self, step=10, nBS=19, isd=500, hbs=25, hue=1.5, default_building_height=15, bmp=None, out="."):
    self.nBS = nBS
    self.isd = isd
    self.step = step
    self.center = [0, 0, 0]
    self.sumo_center = [0, 0, 0]
    self.hbs = hbs
    self.hue = hue
    self.bs = ENB(hbs)
    self.ue = UE(hue)
    if bmp != None:
      self.sumo_center = bmp[1]
    self.snr_calculator = SNRCalculator(self, bmp, self.bs, self.ue, building_height=default_building_height)
    self.deployBSs()
    self.out = out
    if not os.path.exists(out):
      os.makedirs(out)
    self.cells = {}

  def get_map_limits(self):
    if self.nBS == 1:
      x_lim = y_lim = self.isd / 2
    elif self.nBS == 7:
      x_lim = math.sqrt(3) * self.isd / 2 + self.isd / 2
      y_lim = 3 * self.isd / 2
    elif self.nBS == 21:
      x_lim = 3 * math.sqrt(3) * self.isd / 2
      y_lim = 5 * self.isd / 2
    # return [-x_lim + self.sumo_center[0], x_lim + self.sumo_center[0]], \
    #        [-y_lim + self.sumo_center[1], y_lim + self.sumo_center[1]]

    return [-x_lim, x_lim], \
           [-y_lim, y_lim]

  def deployBSs(self):
    h = self.hbs
    assert self.nBS == 1 or self.nBS == 7 or self.nBS == 19

    bsp = []

    bsp.append([0, 0, h])

    if self.nBS == 1:
      self.BSlist = bsp
      return

    bsp.append([0, self.isd, h])
    bsp.append([0, -self.isd, h])
    bsp.append([-math.sqrt(3) * self.isd / 2, self.isd / 2, h])
    bsp.append([-math.sqrt(3) * self.isd / 2, -self.isd / 2, h])
    bsp.append([math.sqrt(3) * self.isd / 2, self.isd / 2, h])
    bsp.append([math.sqrt(3) * self.isd / 2, -self.isd / 2, h])

    if self.nBS == 7:
      self.BSlist = bsp
      return

    bsp.append([math.sqrt(3) * self.isd, 0, h])
    bsp.append([math.sqrt(3) * self.isd, self.isd, h])
    bsp.append([math.sqrt(3) / 2 * self.isd, 3 / 2 * self.isd, h])
    bsp.append([0, 2 * self.isd, h])
    bsp.append([-math.sqrt(3) / 2 * self.isd, 3 / 2 * self.isd, h])
    bsp.append([-math.sqrt(3) * self.isd, self.isd, h])
    bsp.append([-math.sqrt(3) * self.isd, 0, h])
    bsp.append([-math.sqrt(3) * self.isd, -self.isd, h])
    bsp.append([-math.sqrt(3) / 2 * self.isd, -3 / 2 * self.isd, h])
    bsp.append([0, -2 * self.isd, h])
    bsp.append([math.sqrt(3) / 2 * self.isd, -3 / 2 * self.isd, h])
    bsp.append([math.sqrt(3) * self.isd, -self.isd, h])

    self.BSlist = bsp
    return

  #  def setCenter(self,center_p):
  #    self.center = center_p
  #    for bsp in self.BSlist:
  #      for i in range(0,bsp):
  #        bsp[i] += center_p[i]

  def addObstacle(self, obs):
    self.snr_calculator.addObstacle(obs)

  def get_shadow_base_map(self, x_min, x_max, y_min, y_max, step):
    xs = np.arange(x_min, x_max, step)
    ys = np.arange(y_min, y_max, step)
    mesh = np.array(np.meshgrid(xs, ys))
    xys = mesh.T.reshape(-1, 2)
    for i in range(len(xys)):
      xys[i] = Util.reference_position(xys[i], self.step, self.center)

    dcorr = 10
    # corr_fun = (lambda i, j: np.exp(-np.sqrt((xys[int(i)][0]-xys[int(j)][0])**2+(xys[int(i)][1]-xys[int(j)][1]))/dcorr))
    corr_fun = (lambda a, b: np.exp(-np.sqrt(np.sum((xys[a] - xys[b]) ** 2, 1)) / dcorr))
    # corr_m = np.fromfunction(corr_fun, (len(xys), len(xys)), dtype=int)
    corr_m = np.zeros((len(xys), len(xys)))
    for i in range(len(xys)):
      for j in range(len(xys)):
        corr_m = corr_fun(i, j)
    c_m = scipy.linalg.cholesky(corr_m, lower=True)
    x = scipy.stats.norm.rvs(size=(len(xys), 1))
    y = np.dot(c_m, x)
    shadow_base = {}
    for i in range(len(xys)):
      # tuple is hashable, np.array is not
      shadow_base[tuple(xys[i])] = y[i]
    return shadow_base

  def run(self, direction="downlink"):
    limits = self.get_map_limits()
    x_min = int(self.center[0] + limits[0][0])
    x_max = int(self.center[0] + limits[0][1])
    y_min = int(self.center[1] + limits[1][0])
    y_max = int(self.center[1] + limits[1][1])
    lobes = [-120, 0, 120]
    snr_per_pos = {}
    max_snr_per_pos = {}
    # shadow_base = self.get_shadow_base_map(x_min, x_max, y_min, y_max, self.step)
    i = 0
    c = 0
    max_distance = math.sqrt(3) * self.isd / 2
    for b in self.BSlist:
      for lobe in lobes:
        f = open(self.out + "/map-cell_" + str(c) + "-bs_" + str(i) + "-lobe_" + str(lobe) + ".out", "w")
        for xa in np.arange(x_min, x_max, self.step):
          for ya in np.arange(y_min, y_max, self.step):
            (x, y) = Util.reference_position((xa, ya), self.step, self.center)
            if x not in snr_per_pos:
              snr_per_pos[x] = {}
              max_snr_per_pos[x] = {}
            if y not in snr_per_pos[x]:
              snr_per_pos[x][y] = {}
            if math.sqrt((x - b[0]) ** 2 + (y - b[1]) ** 2) > max_distance:
              continue
            if direction == "downlink":
              snr = self.snr_calculator.getSNR(b, [x, y, self.ue.height], lobe, "downlink")
            else:
              snr = self.snr_calculator.getSNR([x, y, self.bs.height], b, lobe, "uplink")
            # snr[0] = snr[0] + snr[1]*np.random.normal()#shadow_base[(x,y)]
            print(x, y, snr[0], snr[1], snr[2], snr[3], sep=",", file=f)
            snr_per_pos[x][y][c] = snr
        f.close()
        self.cells[c] = {'pos': b, 'lobe': lobe, 'id': c}
        c += 1
      i += 1
    f = open(self.out + "/cells.pkl", "wb")
    pickle.dump(self.cells, f)
    f.close()
    f = open(self.out + "/snr_per_pos.pkl", "wb")
    pickle.dump(snr_per_pos, f)
    f.close()
    f = open(self.out + "/map-max.out", "w")
    for xa in np.arange(x_min, x_max, self.step):
      for ya in np.arange(y_min, y_max, self.step):
        (x, y) = Util.reference_position((xa, ya), self.step, self.center)
        if len(snr_per_pos[x][y]) == 0:
          continue
        max_snr = -9999
        max_cell = -1
        for c in snr_per_pos[x][y]:
          if max_cell == -1 or snr_per_pos[x][y][c][0] > max_snr:
            max_snr = snr_per_pos[x][y][c][0]
            max_cell = c
        max_sf = snr_per_pos[x][y][max_cell][1]
        max_los = snr_per_pos[x][y][max_cell][2]
        max_h = snr_per_pos[x][y][max_cell][3]

        max_snr_per_pos[x][y] = {
          'x': x,
          'y': y,
          'cell': max_cell,
          'snr': max_snr,
          'sf': max_sf,
          'los': max_los,
          'h': max_h,
          'bs': self.cells[max_cell]['pos'],
          'lobe': self.cells[max_cell]['lobe']
        }

        print(x, y, max_snr, max_cell, max_sf, max_los, max_h, sep=",", file=f)
    f.close()
    f = open(self.out + "/max_snr_per_pos.pkl", "wb")
    pickle.dump(max_snr_per_pos, f)
    f.close()

    self.snr_calculator.clearObstacles()
    exprt = {
      'x_min': x_min,
      'y_min': y_min,
      'x_max': x_max,
      'y_max': y_max,
      'max_distance': max_distance,
      'snr_calculator': self.snr_calculator,
      'precision': self.step,
      'center': self.center,
      'sumo_center': self.sumo_center
    }
    fconfig = open(self.out + "/sim.config", "wb")
    pickle.dump(exprt, fconfig)
    fconfig.close()


if __name__ == '__main__':

  bfile = '~/sumo_simulations/small-berlin/osm.poly.xml'
  center = [1080.0, 892.0]
  out="small-berlin-long-out"
  nBS = 7

  if len(sys.argv) > 1:
    bfile = sys.argv[1]
  if len(sys.argv) > 2:
    xc = float(sys.argv[2])
    yc = float(sys.argv[3])
    center = [xc, yc]
  if len(sys.argv) > 4:
    out = sys.argv[4]
  if len(sys.argv) > 5:
    nBS = int(sys.argv[5])

  sim = Simulation(step=10, nBS=nBS, bmp=(bfile, center), out=out)
  sim.run()
