import numpy as np
from scipy import fft


class Util:

  @staticmethod
  def reference_position(pos, prec, start=(0, 0)):
    xa = pos[0] - start[0]
    ya = pos[1] - start[1]
    x = xa - (xa % prec) + start[0] + prec / 2
    y = ya - (ya % prec) + start[1] + prec / 2
    return np.array((x, y))

  @staticmethod
  def reference_time(time, prec, start=0):
    ta = round(time * 1000) - round(start * 1000)
    t = ta - (ta % round(prec * 1000)) + round(start * 1000) + round(prec * 1000) / 2
    return int(t)

  @staticmethod
  def freq_domain_view(data, f_s=1, fft_type='double'):
    nfft = int(round(2**np.ceil(np.log2(np.abs(len(data))))))
    if fft_type == 'single':
      signal = fft.fft(data, nfft)
      signal = signal[0:nfft/2]
      f_vals = f_s*np.arange(0, nfft/2)/nfft
    else:
      signal = fft.fftshift(fft.fft(data, nfft))
      f_vals = f_s*np.arange(-nfft/2, nfft/2)/nfft

    return signal, f_vals
