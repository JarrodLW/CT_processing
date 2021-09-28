# Here we do non-blind deconvolution using the square of the probe modulus as kernel, with automatic registration using PALM.

import odl
import numpy as np
from dTV.Ptycho_XRF_project.misc import Embedding, ConvolutionViaFFT, Convolution, PALM
import matplotlib.pyplot as plt
from PIL import Image
from skimage.util import compare_images
import dTV.myFunctionals as fctls

# grabbing data
XRF_image = np.load('dTV/CT_data/Ptycho_XRF_07042021/XRF_W_La.npy')
ptycho = np.load('dTV/CT_data/Ptycho_XRF_07042021/Ptycho.npy')  # just for reference
probe_modulus = np.load('dTV/CT_data/Ptycho_XRF_07042021/probe_modulus.npy')

upsample_factor = 2
conv_impl = 'signal'

kernel_width = int(np.around(upsample_factor*probe_modulus.shape[0]*27.18/100))
margin_width = int(np.around(probe_modulus.shape[0]*27.18/100))
height = int(upsample_factor*XRF_image.shape[0]) + 2*kernel_width
width = int(upsample_factor*XRF_image.shape[1]) + 2*kernel_width

# let's try padding the data
data_arr = np.ones((XRF_image.shape[0]+2*margin_width, XRF_image.shape[0]+2*margin_width))
data_arr[margin_width:-margin_width, margin_width:-margin_width] = XRF_image/np.amax(XRF_image)

# building forward model
image_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
                                          shape=[height, width], dtype='float')

data_height, data_width = data_arr.shape
data_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
                                          shape=[data_height, data_width], dtype='float')

kernel_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
                                          shape=[kernel_width, kernel_width], dtype='float')

# grabbing kernel

probe_modulus_im = Image.fromarray(probe_modulus)
probe_modulus_downsampled = np.array(probe_modulus_im.resize((kernel_width, kernel_width), Image.BICUBIC))

kernel = kernel_space.element(np.flipud(np.fliplr(probe_modulus_downsampled**2)))
#kernel /= kernel.ufuncs.sum()

# the datafit and regularisers
data = data_space.element(data_arr)

if upsample_factor==1:
    ptycho_window = ptycho[362:-361, 362:-361]

elif upsample_factor==2:
    #ptycho_window = ptycho[368:-355, 358:-365]
    #ptycho_window = ptycho[364:-359, 362:-361]
    ptycho_window = ptycho[366:-357, 358:-365]

ptycho_im = Image.fromarray(ptycho_window/np.amax(ptycho_window))
ptycho_downsampled_1 = np.array(ptycho_im.resize((int(upsample_factor * XRF_image.shape[0]),
                                                int(upsample_factor * XRF_image.shape[1])), Image.BICUBIC))
#ptycho_downsampled_2 = np.array(ptycho_im.resize((XRF_image.shape[0], XRF_image.shape[1]), Image.BICUBIC))

sinfo = np.ones((height, width))
sinfo[kernel_width:-kernel_width, kernel_width:-kernel_width] = ptycho_downsampled_1

if upsample_factor!=1: #TODO This is a hack - the sampling operator used below must take the transpose
    sinfo=sinfo.T

# check that guide image is registered
if upsample_factor==1:
    example_deblurred_im = np.load('dTV/Ptycho_XRF_project/Results/non_blind_deblurred_wheel_l1.npy')

elif upsample_factor==2:
    example_deblurred_im = np.load('dTV/Ptycho_XRF_project/Results/non_blind_deblurred_wheel_l1_up_2.npy')
    # example_deblurred_im_padded = np.ones((height, width))
    # example_deblurred_im_padded[kernel_width:-kernel_width, kernel_width:-kernel_width] = np.load('dTV/Ptycho_XRF_project/Results/non_blind_deblurred_wheel_l1.npy')

checkerboard_1 = compare_images(example_deblurred_im, ptycho_downsampled_1, method='checkerboard', n_tiles=(16, 16))

#example_deblurred_im_non_cropped = np.load('dTV/Ptycho_XRF_project/Results/non_blind_deblurred_wheel_l1_non_cropped.npy')
#checkerboard_2 = compare_images(example_deblurred_im_padded, sinfo, method='checkerboard', n_tiles=(16, 16))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle('Checking alignment of guide image')
ax1.imshow(example_deblurred_im, cmap=plt.cm.gray)
ax2.imshow(ptycho_downsampled_1, cmap=plt.cm.gray)
ax3.imshow(checkerboard_1, cmap=plt.cm.gray)

# fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
# fig.suptitle('Checking alignment of guide image')
# ax1.imshow(example_deblurred_im_padded, cmap=plt.cm.gray)
# ax2.imshow(sinfo, cmap=plt.cm.gray)
# ax3.imshow(checkerboard_2, cmap=plt.cm.gray)

#alpha = 0.1
alpha = 0.001
eta = 0.01
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

## the forward operator
if conv_impl=='signal':
    conv = ConvolutionViaFFT(image_space, kernel)

elif conv_impl=='ndimage':
    conv = Convolution(image_space, kernel)

if upsample_factor == 1:
    forward_op = conv

else:
    vert_ind = np.sort(list(int(upsample_factor)*np.arange(data_height))*int(data_width))
    horiz_ind = list(int(upsample_factor)*np.arange(data_width))*int(data_height)
    sampling_points = [vert_ind, horiz_ind]
    emb = Embedding(data_space, image_space, sampling_points=sampling_points, adjoint=None)
    forward_op = emb.adjoint*conv

op_norm = 1.1 * odl.power_method_opnorm(forward_op)
tau = 1.0 / op_norm  # Step size for the primal variable
sigma = 1.0 / op_norm  # Step size for the dual variable

# trying to do non-blind deconvolution

Yaff = odl.tensor_space(6)
X = odl.ProductSpace(image_space, Yaff)
reg_affine = odl.solvers.ZeroFunctional(Yaff)
datafit = fctls.DataFitL2Disp(X, data, forward_op)
Reg = odl.solvers.SeparableSum(reg_image, reg_affine)

x0 = X.element([forward_op.adjoint(data), X[1].zero()])

niter = 100
st = 10
#function_value = datafit + reg_image
cb = (odl.solvers.CallbackPrintIteration(fmt='iter:{:4d}', step=st, end=', ') &
      odl.solvers.CallbackPrintTiming(fmt='time: {:5.2f} s', cumulative=True, step=st, end=', ') &
      odl.solvers.CallbackShow(step=10)) # &
      #odl.solvers.CallbackPrint(function_value, fmt='f(x)={0:.4g}', step=st))

L = [1e4, 1e4]
ud_vars = [0, 1]

palm = PALM(datafit, Reg, ud_vars=ud_vars, x=x0.copy(), L=None, niter=50, callback=cb)

## plotting results

if upsample_factor==1:
    recon = palm.x[0].asarray()

else:
    recon = palm.x[0].asarray().T

discr = np.abs(forward_op(palm.x[0]).asarray()[margin_width:-margin_width, margin_width:-margin_width] - XRF_image/np.amax(XRF_image))

fig, (ax2, ax3, ax4, ax5) = plt.subplots(1, 4)
fig.suptitle('Results')
#im_1 = ax1.imshow(sinfo.T, cmap=plt.cm.gray)
im_2 = ax2.imshow(recon, cmap=plt.cm.gray)
im_3 = ax3.imshow(recon[kernel_width:-kernel_width, kernel_width:-kernel_width], cmap=plt.cm.gray, vmax=0.5)
im_4 = ax4.imshow(XRF_image/np.amax(XRF_image), cmap=plt.cm.gray)
im_5 = ax5.imshow(discr, cmap=plt.cm.gray)
fig.colorbar(im_4, ax=ax4)
fig.colorbar(im_5, ax=ax5)

convolved_sinfo = forward_op(image_space.element(sinfo)).asarray()
convolved_sinfo_rescaled = (convolved_sinfo - np.amin(convolved_sinfo))/(np.amax(convolved_sinfo) - np.amin(convolved_sinfo))
checkerboard_3 = compare_images(convolved_sinfo_rescaled, data.asarray(), method='checkerboard', n_tiles=(16, 16))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(convolved_sinfo_rescaled, cmap=plt.cm.gray)
ax2.imshow(data.asarray(), cmap=plt.cm.gray)
ax3.imshow(checkerboard_3, cmap=plt.cm.gray)
