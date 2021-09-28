import odl
import numpy as np
#from dTV.Ptycho_XRF_project.misc import *
from dTV.Ptycho_XRF_project.misc import DataFitL2LinearPlusConv, DataFitL2LinearPlusConvViaFFT, TotalVariationNonNegative, PALM, Embedding,  ConvolutionViaFFT
from odl.solvers import L2NormSquared as odl_l2sq
from scipy.ndimage import gaussian_filter
import h5py
import matplotlib.pyplot as plt
from skimage import restoration
from PIL import Image
from scipy.signal import convolve as signal_convolve
from scipy.ndimage import convolve as sp_convolve

mode = 'non-blind deblur'

# grabbing data
XRF_image = np.load('dTV/CT_data/Ptycho_XRF_07042021/XRF_W_La.npy')
ptycho = np.load('dTV/CT_data/Ptycho_XRF_07042021/Ptycho.npy')
probe_modulus = np.load('dTV/CT_data/Ptycho_XRF_07042021/probe_modulus.npy')

# seeing what the convolution against the squared probe modulus looks like
convolved_image = sp_convolve(ptycho, probe_modulus**2, mode='constant')

plt.figure()
plt.imshow(convolved_image, cmap=plt.cm.gray)

plt.figure()
plt.imshow(convolved_image[372:-351, 369:-354], cmap=plt.cm.gray)

plt.figure()
plt.imshow(XRF_image, cmap=plt.cm.gray)
plt.colorbar()

#

ptycho_window = ptycho[372:-351, 369:-354]

plt.figure()
plt.imshow(ptycho_window, cmap=plt.cm.gray)


upsample_factor = 1.
kernel_width = int(np.around(upsample_factor*probe_modulus.shape[0]*27.18/100))

# let's try padding the data
data_arr = np.ones((XRF_image.shape[0]+2*kernel_width, XRF_image.shape[0]+2*kernel_width))
data_arr[kernel_width:-kernel_width, kernel_width:-kernel_width] = XRF_image/np.amax(XRF_image)

# building forward model
height = int(upsample_factor*XRF_image.shape[0])
width = int(upsample_factor*XRF_image.shape[1])
# image_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
#                                           shape=[height, width], dtype='float')

image_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
                                          shape=[data_arr.shape[0], data_arr.shape[1]], dtype='float')

#data_height, data_width = XRF_image.shape
data_height, data_width = data_arr.shape
data_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
                                          shape=[data_height, data_width], dtype='float')

#kernel_height, kernel_width = probe_modulus.shape
# kernel_height, kernel_width = (7, 7)
kernel_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
                                          shape=[kernel_width, kernel_width], dtype='float')

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
lambda_image = 0.000001
TV_image = TotalVariationNonNegative(image_space, alpha=lambda_image)

if mode=='kernel estimation':
    kernel_reg = odl.solvers.IndicatorNonnegativity(kernel_space)
    #lambda_kernel = 0.01
    #kernel_reg = TotalVariationNonNegative(kernel_space, alpha=lambda_kernel)

elif mode=='blind deblur':
    kernel_reg = odl.solvers.IndicatorSimplex(kernel_space)

elif mode=='non-blind deblur':
    kernel_reg = odl.solvers.ZeroFunctional(kernel_space)

# the datafit, with convolution

if upsample_factor == 1:
    linear_op = odl.IdentityOperator(image_space)

else: # I think vert and horiz indices need to be swapped ---see Nonblind_deconvolution_XRF_with_PDHG
    vert_ind = np.sort(list(int(upsample_factor)*np.arange(data_height))*int(data_width))
    horiz_ind = list(int(upsample_factor)*np.arange(data_width))*int(data_height)
    sampling_points = [vert_ind, horiz_ind]
    emb = Embedding(data_space, image_space, sampling_points=sampling_points, adjoint=None)
    linear_op = emb.adjoint

#data = data_space.element(XRF_image/np.amax(XRF_image))
data = data_space.element(data_arr)

# ptycho image
im = Image.fromarray(ptycho_window)
# new_image = np.array(im.resize(size, PIL.Image.BICUBIC))
ptycho_resized = np.array(im.resize((width, height)))
ptycho_resized -= np.amin(ptycho_resized)
#ptycho_odl = image_space.element(ptycho_resized)

domain = odl.space.pspace.ProductSpace(image_space, kernel_space)
#datafit = DataFitL2LinearPlusConvViaFFT(domain, linear_op, data)
datafit = DataFitL2LinearPlusConv(domain, linear_op, data)

Reg = odl.solvers.SeparableSum(TV_image, kernel_reg)

# trying to do blind deconvolution

# initialisation for blind deconvolution

if mode=='blind deblur':

    kernel_init = gkern(kernel_height, sig=1)

    recon_init = (kernel_height*kernel_width/kernel_space.domain.volume)*restoration.richardson_lucy(data.asarray(), kernel_init, iterations=30)
    im = Image.fromarray(XRF_image/np.amax(XRF_image))
    recon_init = np.array(im.resize((int(upsample_factor)*XRF_image.shape[0], int(upsample_factor)*XRF_image.shape[0]), Image.BICUBIC))

    ud_vars = [0, 1]

# initialisation for non-blind deblurring

elif mode=='non-blind deblur':

    recon_init = image_space.zero()
    #recon_init = 2500*ptycho_odl # prefactor is fudged
    #kernel_init = kernel_space.element(probe_modulus**2)

    probe_modulus_im = Image.fromarray(probe_modulus)
    probe_modulus_downsampled = np.array(probe_modulus_im.resize((kernel_width, kernel_width), Image.BICUBIC))

    kernel_init = kernel_space.element(np.flipud(np.fliplr(probe_modulus_downsampled**2)))
    kernel_init /= kernel_init.ufuncs.sum()

    ud_vars = [0]

# initialisation for kernel estimation

#recon_init = 2.5*1e3*ptycho_odl
elif mode=='kernel estimation':

    recon_init = ptycho_odl
    kernel_init = gkern(kernel_width, sig=5)

    ud_vars = [1]

#recon_init = (1/upsample_factor**2)*forward_op.adjoint(data)
x = domain.element([recon_init, kernel_init])


L = [1e4, 1e4]

st = 1
function_value = datafit + Reg
cb = (odl.solvers.CallbackPrintIteration(fmt='iter:{:4d}', step=st, end=', ')  &
      odl.solvers.CallbackPrintTiming(fmt='time: {:5.2f} s', cumulative=True, step=st, end=', ') &
      odl.solvers.CallbackShow(step=5) &
      odl.solvers.CallbackPrint(function_value, fmt='f(x)={0:.4g}', step=st))

PALM(datafit, Reg, ud_vars=ud_vars, x=x, L=None, niter=500, callback=cb)

# x.show()

conv = ConvolutionViaFFT(image_space, x[1])
conv(x[0]).show()

data.show()

(conv(x[0]) - data).show()