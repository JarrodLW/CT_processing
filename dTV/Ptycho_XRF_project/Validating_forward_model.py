# In this script, I initialise the reconstruction at the rescaled, registered pytchographic recon,
# and perform updates only only the kernel. The aim is to see how well the problem is modelled by
# as a de-blurring problem. We're assuming that the ptychographic reconstruction can be considered a
# reasonable ground-truth for XRF in this case.

import odl
import numpy as np
from dTV.Ptycho_XRF_project.misc import *
from odl.solvers import L2NormSquared as odl_l2sq
import matplotlib.pyplot as plt
import scipy as sp
from PIL import Image
from scipy.signal import convolve as signal_convolve
from scipy.ndimage import convolve as sp_convolve
from time import time

# grabbing data
XRF_image = np.load('dTV/CT_data/Ptycho_XRF_07042021/XRF_W_La.npy')
ptycho = np.load('dTV/CT_data/Ptycho_XRF_07042021/Ptycho.npy')
probe_modulus = np.load('dTV/CT_data/Ptycho_XRF_07042021/probe_modulus.npy')

# testing convolutions
t0 = time()
sp_convolved_image = sp_convolve(ptycho, probe_modulus[::-1, ::-1]**2, mode='constant')
t1 = time()
print("sp_convolve done in "+str(t1-t0))

t0 = time()
signal_convolved_image = signal_convolve(ptycho, probe_modulus[::-1, ::-1]**2, mode='same')
t1 = time()
print("signal_convolve done in "+str(t1-t0))

plt.figure()
plt.imshow(sp_convolved_image, cmap=plt.cm.gray)

plt.figure()
plt.imshow(sp_convolved_image[362:-361, 369:-354], cmap=plt.cm.gray)

plt.figure()
plt.imshow(signal_convolved_image, cmap=plt.cm.gray)

plt.figure()
plt.imshow(signal_convolved_image[372:-351, 369:-354], cmap=plt.cm.gray)

plt.figure()
plt.imshow(XRF_image, cmap=plt.cm.gray)
plt.colorbar()



# modifying field-of-view of ptycho and rescaling pixel intensity

ptycho_window = ptycho[372:-351, 369:-354]

im = Image.fromarray(ptycho_window)
# new_image = np.array(im.resize(size, PIL.Image.BICUBIC))
ptycho_resized = np.array(im.resize(XRF_image.shape))
ptycho_resized -= np.amin(ptycho_resized)

# plt.figure()
# plt.imshow(XRF_image, cmap=plt.cm.gray)
#
# plt.figure()
# plt.imshow(ptycho_resized, cmap=plt.cm.gray)


# building forward model
height, width = XRF_image.shape
image_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
                                          shape=[height, width], dtype='float')

kernel_height = 5
kernel_width = 1
kernel_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
                                          shape=[kernel_height, kernel_width], dtype='float')

# gaussian kernel for initialisation
def gkern(l, sig=1.):
    """\
    creates gaussian kernel with side length l and a sigma of sig
    """

    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

    return kernel / np.sum(kernel)


# regularisers
TV_image = odl.solvers.ZeroFunctional(image_space)
kernel_reg = odl.solvers.IndicatorNonnegativity(kernel_space)

# the datafit, with convolution
lin_op = odl.IdentityOperator(image_space)
data = lin_op.range.element(XRF_image/np.amax(XRF_image))
domain = odl.space.pspace.ProductSpace(image_space, kernel_space)
datafit = DataFitL2LinearPlusConv(domain, lin_op, data)

Reg = odl.solvers.SeparableSum(TV_image, kernel_reg)

#kernel_init = gkern(kernel_height, sig=1.5)
kernel_init = np.zeros((kernel_height, kernel_width))
recon_init = image_space.element(ptycho_resized)
x = domain.element([recon_init.copy(), kernel_init.copy()])

ud_vars = [1]
#L = [1, 1e-3]

st = 1
function_value = datafit + Reg
cb = (odl.solvers.CallbackPrintIteration(fmt='iter:{:4d}', step=st, end=', ') &
      odl.solvers.CallbackPrintTiming(fmt='time: {:5.2f} s', cumulative=True, step=st, end=', ') &
      odl.solvers.CallbackShow(step=5) &
      odl.solvers.CallbackPrint(function_value, fmt='f(x)={0:.4g}', step=st))

PALM(datafit, Reg, ud_vars=ud_vars, x=x, L=None, niter=500, callback=cb)

conv = Convolution(image_space, x[1])
conv(recon_init).show(title='Blurred ptycho')
data.show(title='Data')


#x.show()

#kernel_init = np.ones((kernel_height, kernel_width))
#kernel_init[1:-1, 1:-1] = 0
kernel_init = np.zeros((kernel_height, kernel_width))
kernel_init[-1, kernel_width//2] = 0.6
kernel_init[kernel_height//2, kernel_width//2] = 1
conv_init = Convolution(image_space, kernel_space.element(kernel_init))
conv_init(recon_init).show()
data.show()
kernel_space.element(kernel_init).show()

# odl.solvers.L2NormSquared(image_space)(conv(recon_init) - data)

#
probe_modulus = np.load('dTV/CT_data/Ptycho_XRF_07042021/probe_modulus.npy')

height, width = ptycho.shape
kernel_height, kernel_width = probe_modulus.shape
image_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
                                          shape=[height, width], dtype='float')

kernel_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
                                          shape=[kernel_height, kernel_width], dtype='float')

kernel = kernel_space.element(probe_modulus**2/np.sum(probe_modulus**2))
kernel.show()
#conv = Convolution(image_space, kernel)
conv = Convolution(image_space, kernel_space.element(probe_modulus))

conv(image_space.element(ptycho)).show()

from scipy.ndimage import convolve as sp_convolve
from scipy.signal import fftconvolve
from time import time

# sp_convolve(ptycho, probe_modulus**2/np.sum(probe_modulus**2))
impl = 'ndimage'

if impl=='fft':
    convolve = fftconvolve

elif impl=='ndimage':
    convolve = sp_convolve

t0 = time()
convolved = convolve(ptycho, probe_modulus**2)
dt = time() - t0
print('done in %.2fs.' % dt)

t0 = time()
convolved_1 = convolve(np.ones((164, 164)), np.ones(probe_modulus.shape))
dt = time() - t0
print('done in %.2fs.' % dt)

t0 = time()
convolved_2 = convolve(np.ones((164, 164)), np.ones((5, 5)))
dt = time() - t0
print('done in %.2fs.' % dt)

plt.figure()
plt.imshow(convolved, cmap=plt.cm.gray)
plt.colorbar()
