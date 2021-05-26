import numpy as np
from scipy import signal
from skimage.measure import compare_ssim

def trans2Y(img):
  img_r = img[:, 0, :, :]
  img_g = img[:, 1, :, :]
  img_b = img[:, 2, :, :]
  img_y = 0.256789 * img_r + 0.504129 * img_g + 0.097906 * img_b + 16
  return img_y

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
  """
  2D gaussian mask - should give the same result as MATLAB's fspecial('gaussian',[shape],[sigma])
  Acknowledgement : https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python (Author@ali_m)
  """
  m,n = [(ss-1.)/2. for ss in shape]
  y,x = np.ogrid[-m:m+1,-n:n+1]
  h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
  h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
  sumh = h.sum()
  if sumh != 0:
    h /= sumh
  return h

def calc_ssim(X, Y, scale, dataset=None, sigma=1.5, K1=0.01, K2=0.03, R=255):
  '''
  X : y channel (i.e., luminance) of transformed YCbCr space of X
  Y : y channel (i.e., luminance) of transformed YCbCr space of Y
  Please follow the setting of psnr_ssim.m in EDSR (Enhanced Deep Residual Networks for Single Image Super-Resolution CVPRW2017).
  Official Link : https://github.com/LimBee/NTIRE2017/tree/db34606c2844e89317aac8728a2de562ef1f8aba
  The authors of EDSR use MATLAB's ssim as the evaluation tool,
  thus this function is the same as ssim.m in MATLAB with C(3) == C(2)/2.
  '''
  gaussian_filter = matlab_style_gauss2D((11, 11), sigma)

  X = trans2Y(X).squeeze()
  Y = trans2Y(Y).squeeze()
  X = X.cpu().numpy().astype(np.float64)
  Y = Y.cpu().numpy().astype(np.float64)

  shave = scale
  if dataset and not dataset.dataset.benchmark:
      shave = scale + 6
  X = X[shave:-shave, shave:-shave]
  Y = Y[shave:-shave, shave:-shave]

  window = gaussian_filter / np.sum(np.sum(gaussian_filter))

  window = np.fliplr(window)
  window = np.flipud(window)

  ux = signal.convolve2d(X, window, mode='valid', boundary='fill', fillvalue=0)
  uy = signal.convolve2d(Y, window, mode='valid', boundary='fill', fillvalue=0)

  uxx = signal.convolve2d(X * X, window, mode='valid', boundary='fill', fillvalue=0)
  uyy = signal.convolve2d(Y * Y, window, mode='valid', boundary='fill', fillvalue=0)
  uxy = signal.convolve2d(X * Y, window, mode='valid', boundary='fill', fillvalue=0)

  vx = uxx - ux * ux
  vy = uyy - uy * uy
  vxy = uxy - ux * uy

  C1 = (K1 * R) ** 2
  C2 = (K2 * R) ** 2

  A1, A2, B1, B2 = ((2 * ux * uy + C1, 2 * vxy + C2, ux ** 2 + uy ** 2 + C1, vx + vy + C2))
  D = B1 * B2
  S = (A1 * A2) / D
  mssim = S.mean()

  # window = gaussian_filter
  #
  # ux = signal.convolve2d(X, window, mode='same', boundary='symm')
  # uy = signal.convolve2d(Y, window, mode='same', boundary='symm')
  #
  # uxx = signal.convolve2d(X*X, window, mode='same', boundary='symm')
  # uyy = signal.convolve2d(Y*Y, window, mode='same', boundary='symm')
  # uxy = signal.convolve2d(X*Y, window, mode='same', boundary='symm')
  #
  # vx = uxx - ux * ux
  # vy = uyy - uy * uy
  # vxy = uxy - ux * uy
  #
  # C1 = (K1 * R) ** 2
  # C2 = (K2 * R) ** 2
  #
  # A1, A2, B1, B2 = ((2 * ux * uy + C1, 2 * vxy + C2, ux ** 2 + uy ** 2 + C1, vx + vy + C2))
  # D = B1 * B2
  # S = (A1 * A2) / D
  # mssim = S.mean()

  return mssim