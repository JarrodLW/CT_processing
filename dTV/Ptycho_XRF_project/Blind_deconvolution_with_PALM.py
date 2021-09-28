# Blind deconvolution. Created 08/07/2021

# model 1 (non-guided): \Vert K*z - u\Vert_2^2 + TVNN(z) + ind_simplex(K)
# model 2 (non-guided): \Vert K*z - u\Vert_2^2 with linearised Bregman (TV) iterations
# model 3 (guided): \Vert K*z - u\Vert_2^2 + dTVNN(z; v) + ind_simplex(K)

# Models 1 and 3 will be minimised via PALM.

import odl
import numpy as np
from dTV.Ptycho_XRF_project.misc import TotalVariationNonNegative, Embedding, \
    DataFitL2LinearPlusConv, DataFitL2LinearPlusConvViaFFT, PALM, get_central_sampling_points
from PIL import Image
import dTV.myFunctionals as fctls
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

model = "model_3"
conv_impl = 'signal'
upsample_factor = 1

probe_modulus = np.load('dTV/CT_data/Ptycho_XRF_07042021/probe_modulus.npy')
XRF_image = np.load('dTV/CT_data/Ptycho_XRF_07042021/XRF_W_La.npy')
ptycho = np.load('dTV/CT_data/Ptycho_XRF_07042021/Ptycho.npy')  # just for reference

kernel_width = int(np.around(upsample_factor * probe_modulus.shape[0] * 27.18 / 100))
margin_width = int(np.around(probe_modulus.shape[0] * 27.18 / 100))

height = int(upsample_factor * XRF_image.shape[0]) + 2 * kernel_width
width = int(upsample_factor * XRF_image.shape[1]) + 2 * kernel_width

# function for initialising kernel

def gkern(l, sig=1.):
    """\
    creates gaussian kernel with side length l and a sigma of sig
    """

    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

    return kernel / np.sum(kernel)

# fixing spaces

image_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
                                          shape=[height, width], dtype='float')

data_height, data_width = XRF_image.shape
data_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
                                          shape=[data_height, data_width], dtype='float')

kernel_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
                                          shape=[kernel_width, kernel_width], dtype='float')

domain = odl.space.pspace.ProductSpace(image_space, kernel_space)

# padding data
# data_arr = np.ones((XRF_image.shape[0]+2*kernel_width, XRF_image.shape[0]+2*kernel_width))
# data_arr[kernel_width:-kernel_width, kernel_width:-kernel_width] = XRF_image/np.amax(XRF_image)
# data = data_space.element(data_arr)

data = data_space.element(XRF_image/np.amax(XRF_image))

if model=='model_1':

    lambda_image = 0.01
    image_reg = TotalVariationNonNegative(image_space, alpha=lambda_image)
    kernel_reg = odl.solvers.IndicatorSimplex(kernel_space)
    ud_vars = [0, 1]

elif model=='model_3':

    ptycho_window = ptycho[362:-361, 362:-361]
    ptycho_im = Image.fromarray(ptycho_window / np.amax(ptycho_window))
    ptycho_downsampled_1 = np.array(ptycho_im.resize((int(upsample_factor * XRF_image.shape[0]),
                                                      int(upsample_factor * XRF_image.shape[1])), Image.BICUBIC))

    sinfo = np.zeros((height, width))
    sinfo[kernel_width: - kernel_width, kernel_width:- kernel_width] = ptycho_downsampled_1

    alpha = 1.
    eta = 0.001
    gamma = 0.99
    strong_cvx = 1e-5
    niter_prox = 20
    prox_options = {}
    prox_options['name'] = 'FGP'
    prox_options['warmstart'] = True
    prox_options['p'] = None
    prox_options['tol'] = None
    prox_options['niter'] = niter_prox
    image_reg = fctls.directionalTotalVariationNonnegative(image_space, alpha=alpha, sinfo=sinfo,
                                                           gamma=gamma, eta=eta, NonNeg=True,
                                                           strong_convexity=strong_cvx, prox_options=prox_options)

    kernel_reg = odl.solvers.IndicatorSimplex(kernel_space)
    ud_vars = [0, 1]

# building the datafit

if upsample_factor == 1:
    #linear_op = odl.IdentityOperator(image_space)
    sampling_points = get_central_sampling_points(data.shape, image_space.shape)
    emb = Embedding(data_space, image_space, sampling_points=sampling_points, adjoint=None)
    linear_op = emb.adjoint

else:
    vert_ind = np.sort(list(1 + kernel_width + int(upsample_factor) * np.arange(data_height)) * int(data_width))
    horiz_ind = list(1 + kernel_width + int(upsample_factor) * np.arange(data_width)) * int(data_height)
    sampling_points = [horiz_ind, vert_ind]
    emb = Embedding(data_space, image_space, sampling_points=sampling_points, adjoint=None)
    linear_op = emb.adjoint

if conv_impl=='signal':
    #conv = ConvolutionViaFFT(image_space, kernel)
    datafit = DataFitL2LinearPlusConvViaFFT(domain, linear_op, data)

elif conv_impl=='ndimage':
    #conv = Convolution(image_space, kernel)
    datafit = DataFitL2LinearPlusConv(domain, linear_op, data)

Reg = odl.solvers.SeparableSum(image_reg, kernel_reg)

kernel_init = gkern(kernel_width, sig=1)

#recon_init = np.zeros((height, width))
#recon_init[kernel_width:-kernel_width, kernel_width:-kernel_width] = data.asarray()
recon_init = linear_op.adjoint(data).asarray()
x = domain.element([recon_init, kernel_init])

niter = 500
st = 10
#function_value = datafit*linear_op + reg_image
cb = (odl.solvers.CallbackPrintIteration(fmt='iter:{:4d}', step=st, end=', ') &
      odl.solvers.CallbackPrintTiming(fmt='time: {:5.2f} s', cumulative=True, step=st, end=', ') &
      odl.solvers.CallbackShow(step=10))
# &
#       odl.solvers.CallbackPrint(function_value, fmt='f(x)={0:.4g}', step=st) &
#       odl.solvers.CallbackPrint(datafit*linear_op, fmt='datafit={0:.4g}', step=st) &
#       odl.solvers.CallbackPrint(reg_image, fmt='reg={0:.4g}', step=st) &
#       odl.solvers.CallbackShowConvergence(function_value))

PALM(datafit, Reg, ud_vars=ud_vars, x=x, L=None, niter=niter, callback=cb)
