# Here we use PALM to minimise  \Vert K*z - w\Vert^2 + \lambda dTV(z;v) + ind(K)
# in two cases:
#
#  (1) 'direct': \lambda=0, only perform updates on K, ind(K) is the non-negativity indicator
#  (2) 'indirect': \lambda>0, perform updates on both z and K, ind(K) is the indicator for the unit simplex

import odl
import numpy as np
#from dTV.Ptycho_XRF_project.misc import *
from dTV.Ptycho_XRF_project.misc import Embedding, DataFitL2LinearPlusConv, DataFitL2LinearPlusConvViaFFT, ConvolutionViaFFT, \
    Convolution, get_central_sampling_points, PALM
from odl.solvers import L2NormSquared as odl_l2sq
from scipy.ndimage import gaussian_filter
import h5py
import matplotlib.pyplot as plt
from skimage import restoration
from PIL import Image
from scipy.signal import convolve as signal_convolve
from scipy.ndimage import convolve as sp_convolve
from skimage.transform import resize
import dTV.myFunctionals as fctls
from mpl_toolkits.axes_grid1 import make_axes_locatable

#comparison_type = 'direct'
comparison_type = 'indirect'

#data = 'dataset_1'
# dataset = 'dataset_2'
#
# # grabbing data
# if dataset=='dataset_1':
#     XRF_image = np.load('dTV/CT_data/Ptycho_XRF_07042021/XRF_W_La.npy')
#     ptycho = np.load('dTV/CT_data/Ptycho_XRF_07042021/Ptycho.npy')  # just for reference
#     kernel_width = int(np.around(upsample_factor * probe_modulus.shape[0] * 27.18 / 100))
#     margin_width = int(np.around(probe_modulus.shape[0] * 27.18 / 100))
#
# elif dataset=='dataset_2':

phase_contrast_image = np.load('dTV/CT_data/Ptycho_XRF_27042021/phase_contrast.npy')
ptycho = np.load('dTV/CT_data/Ptycho_XRF_27042021/Ptycho.npy')
scaling_factor = 200/27.52
ptycho_downsized = resize(ptycho, (ptycho.shape[0]//scaling_factor, ptycho.shape[1]//scaling_factor))
ptycho_downsized = ptycho_downsized[3:-6, :-8]
kernel_height = kernel_width = 5

margin_width = kernel_width

# plt.figure()
# plt.imshow(ptycho, cmap=plt.cm.gray)

plt.figure()
plt.imshow(ptycho_downsized, cmap=plt.cm.gray)

plt.figure()
plt.imshow(phase_contrast_image, cmap=plt.cm.gray)

upsample_factor = 1
data_fit = "l2sq"
#conv_impl = 'signal'
#reg_type = 'dTVNN'

data_height, data_width = phase_contrast_image.shape
data_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
                                          shape=[data_height, data_width], dtype='float')

kernel_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
                                          shape=[kernel_width, kernel_width], dtype='float')

data = data_space.element((phase_contrast_image - np.amin(phase_contrast_image))/
                          (np.amax(phase_contrast_image) - np.amin(phase_contrast_image)))

if upsample_factor==1:
    height = phase_contrast_image.shape[0] + 2 * kernel_width
    width = phase_contrast_image.shape[1] + 2 * kernel_width

if data_fit=='l2sq':
    datafit = odl.solvers.L2NormSquared(data_space).translated(data)
    lambda_image = 0.00001

image_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
                                          shape=[height, width], dtype='float')

if comparison_type=='direct':

    reg_image == odl.solvers.ZeroFunctional(image_space)
    kernel_reg = odl.solvers.IndicatorNonnegativity(kernel_space)
    ud_vars = [0]

elif comparison_type=='indirect':

    if upsample_factor==1:
        ptycho_window = ptycho_downsized

    sinfo = np.zeros((height, width))
    sinfo[kernel_width: - kernel_width, kernel_width:- kernel_width] = ptycho_window

    alpha = 0.0001
    eta = 0.0001
    gamma = 0.99
    strong_cvx = 1e-5
    niter_prox = 20
    prox_options = {}
    prox_options['name'] = 'FGP'
    prox_options['warmstart'] = True
    prox_options['p'] = None
    prox_options['tol'] = None
    prox_options['niter'] = niter_prox
    reg_image = fctls.directionalTotalVariationNonnegative(image_space, alpha=alpha, sinfo=sinfo,
                                                           gamma=gamma, eta=eta, NonNeg=True,
                                                           strong_convexity=strong_cvx, prox_options=prox_options)

    kernel_reg = odl.solvers.IndicatorSimplex(kernel_space)
    ud_vars = [0, 1]

if upsample_factor == 1:
    #linear_op = odl.IdentityOperator(image_space)
    if upsample_factor == 1:
        sampling_points = get_central_sampling_points(data.shape, image_space.shape)
        emb = Embedding(data_space, image_space, sampling_points=sampling_points, adjoint=None)
        linear_op = emb.adjoint

domain = odl.space.pspace.ProductSpace(image_space, kernel_space)
#datafit = DataFitL2LinearPlusConvViaFFT(domain, linear_op, data)
datafit = DataFitL2LinearPlusConv(domain, linear_op, data)

# initialisation
def gkern(l, sig=1.):
    """\
    creates gaussian kernel with side length l and a sigma of sig
    """

    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

    return kernel / np.sum(kernel)


probe_modulus = np.load('dTV/CT_data/Ptycho_XRF_07042021/probe_modulus.npy')

kernel_init = resize(probe_modulus.T**2, (5, 5))
kernel_init /= np.sum(kernel_init)
#kernel_init = gkern(kernel_height, sig=1)

x = domain.element([sinfo.copy(), kernel_init])

Reg = odl.solvers.SeparableSum(reg_image, kernel_reg)

L = [1e4, 1e4]

st = 1
function_value = datafit + Reg
cb = (odl.solvers.CallbackPrintIteration(fmt='iter:{:4d}', step=st, end=', ')  &
      odl.solvers.CallbackPrintTiming(fmt='time: {:5.2f} s', cumulative=True, step=st, end=', ') &
      odl.solvers.CallbackShow(step=5) &
      odl.solvers.CallbackPrint(function_value, fmt='f(x)={0:.4g}', step=st) &
      odl.solvers.CallbackShowConvergence(function_value))

PALM(datafit, Reg, ud_vars=ud_vars, x=x, L=None, niter=500, callback=cb)

conv = Convolution(image_space, x[1])
blurred_recon = conv(x[0])
blurred_recon_window = blurred_recon.asarray()[5:-5, 5:-5]

recon = x[0].asarray()[kernel_height:-kernel_height, kernel_width:-kernel_width]
kernel = x[1].asarray()

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(15, 3))
im_1 = ax1.imshow(ptycho_window, cmap=plt.cm.gray)
ax1.set_title("ptycho image")
im_2 = ax2.imshow(recon, cmap=plt.cm.gray)
ax2.set_title("recon")
im_3 = ax3.imshow(data.asarray(), cmap=plt.cm.gray)
ax3.set_title("(data) original phase contrast")
im_4 = ax4.imshow(blurred_recon_window, cmap=plt.cm.gray)
ax4.set_title("blurred recon")
im_5 = ax5.imshow(np.abs(data.asarray()-blurred_recon_window), cmap=plt.cm.gray)
ax5.set_title("abs diff")
divider_1 = make_axes_locatable(ax1)
divider_2 = make_axes_locatable(ax2)
divider_3 = make_axes_locatable(ax3)
divider_4 = make_axes_locatable(ax4)
divider_5 = make_axes_locatable(ax5)
cax1 = divider_1.append_axes("right", size="5%", pad=0.04)
cax2 = divider_2.append_axes("right", size="5%", pad=0.04)
cax3 = divider_3.append_axes("right", size="5%", pad=0.04)
cax4 = divider_4.append_axes("right", size="5%", pad=0.04)
cax5 = divider_5.append_axes("right", size="5%", pad=0.04)
plt.colorbar(im_1, cax=cax1)
plt.colorbar(im_2, cax=cax2)
plt.colorbar(im_3, cax=cax3)
plt.colorbar(im_4, cax=cax4)
plt.colorbar(im_5, cax=cax5)
plt.tight_layout(rect=[3, 3, 3, 3])


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(kernel_init, cmap=plt.cm.gray)
ax1.set_title("kernel initialisation")
ax2.imshow(kernel, cmap=plt.cm.gray)
ax2.set_title("estimated kernel")

