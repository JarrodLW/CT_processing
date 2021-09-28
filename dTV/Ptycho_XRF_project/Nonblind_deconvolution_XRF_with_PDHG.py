# Here we do non-blind deconvolution using the square of the probe modulus as kernel. We use PDHG. Registration done manually.

import odl
import numpy as np
from dTV.Ptycho_XRF_project.misc import TotalVariationNonNegative, Embedding, ConvolutionViaFFT, Convolution, get_central_sampling_points
from odl.solvers import L2NormSquared as odl_l2sq
import matplotlib.pyplot as plt
from PIL import Image
# from skimage import restoration
# from scipy.signal import convolve as signal_convolve
# from scipy.ndimage import convolve as sp_convolve
from skimage.util import compare_images
import dTV.myFunctionals as fctls
from mpl_toolkits.axes_grid1 import make_axes_locatable

# grabbing data

#data = 'dataset_1'
dataset = 'dataset_2'
upsample_factor = 1

probe_modulus = np.load('dTV/CT_data/Ptycho_XRF_07042021/probe_modulus.npy')

if dataset=='dataset_1':
    XRF_image = np.load('dTV/CT_data/Ptycho_XRF_07042021/XRF_W_La.npy')
    ptycho = np.load('dTV/CT_data/Ptycho_XRF_07042021/Ptycho.npy')  # just for reference

    kernel_width = int(np.around(upsample_factor * probe_modulus.shape[0] * 27.18 / 100))
    margin_width = int(np.around(probe_modulus.shape[0] * 27.18 / 100))

elif dataset=='dataset_2':
    XRF_image = np.load('dTV/CT_data/Ptycho_XRF_27042021/XRF_Ca_Ka.npy')
    ptycho = np.load('dTV/CT_data/Ptycho_XRF_27042021/Ptycho.npy')
    kernel_height = kernel_width = 5

data_fit = 'l2sq'
#data_fit = 'l1'
#data_fit = 'KL'
conv_impl = 'signal'
reg_type = 'TVNN'

# let's try data of dimension 160x160
#XRF_image = XRF_image[1:, 1:]

if upsample_factor==1: #TODO no need for if statements. Remove them.
    height = XRF_image.shape[0] + 2 * kernel_width
    width = XRF_image.shape[1] + 2 * kernel_width

elif upsample_factor==2:
    height = int(upsample_factor * XRF_image.shape[0]) + 2 * kernel_width
    width = int(upsample_factor * XRF_image.shape[1]) + 2 * kernel_width

elif upsample_factor==3:
    height = int(upsample_factor * XRF_image.shape[0]) + 2 * kernel_width
    width = int(upsample_factor * XRF_image.shape[1]) + 2 * kernel_width

# let's try padding the data
# data_arr = np.ones((XRF_image.shape[0]+2*margin_width, XRF_image.shape[0]+2*margin_width))
# data_arr[margin_width:-margin_width, margin_width:-margin_width] = XRF_image/np.amax(XRF_image)

# building forward model
# image_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
#                                           shape=[height, width], dtype='float')

image_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
                                          shape=[height, width], dtype='float')

data_height, data_width = XRF_image.shape
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
#data = data_space.element((XRF_image/np.amax(XRF_image).T))

if dataset=='dataset_1':
    data = data_space.element(XRF_image/np.amax(XRF_image))

elif dataset=='dataset_2':
    data = data_space.element(XRF_image//1)

if data_fit=='l1':
    datafit = odl.solvers.L1Norm(data_space).translated(data)
    lambda_image = 0.001
    #lambda_image = 0.003

elif data_fit=='l2sq':
    datafit = odl.solvers.L2NormSquared(data_space).translated(data)
    #lambda_image = 0.00001 # for dataset_1
    lambda_image = 0.03 # for dataset_2

elif data_fit=='KL':
    data = np.round(data)
    datafit = odl.solvers.functional.default_functionals.KullbackLeibler(data_space, prior=data)
    lambda_image = 0.00001

print(f"reg={reg_type}: data={data}: upsample_factor={upsample_factor}")

if reg_type=='TVNN':
    reg_image = TotalVariationNonNegative(image_space, alpha=lambda_image)

elif reg_type=='dTVNN':

    if dataset=='dataset_1':

        if upsample_factor==1:
            ptycho_window = ptycho[362:-361, 362:-361]

        elif upsample_factor==2:
            ptycho_window = ptycho[362:-361, 362:-361]
            #ptycho_window = ptycho[368:-355, 358:-365]
            #ptycho_window = ptycho[369:-357, 361:-365]

        elif upsample_factor==3:
            #ptycho_window = ptycho[366:-357, 358:-365]
            ptycho_window = ptycho[362:-361, 362:-361]

        ptycho_im = Image.fromarray(ptycho_window/np.amax(ptycho_window))
        ptycho_downsampled_1 = np.array(ptycho_im.resize((int(upsample_factor * XRF_image.shape[0]),
                                                        int(upsample_factor * XRF_image.shape[1])), Image.BICUBIC))
        #ptycho_downsampled_2 = np.array(ptycho_im.resize((XRF_image.shape[0], XRF_image.shape[1]), Image.BICUBIC))

        #sinfo = 0.42*np.ones((height, width))
        sinfo = np.zeros((height, width))
        sinfo[kernel_width: - kernel_width, kernel_width:- kernel_width] = ptycho_downsampled_1

        # check that guide image is registered
        if upsample_factor==1:
            example_deblurred_im = np.load('dTV/Ptycho_XRF_project/Results/non_blind_deblurred_wheel_l1_non_cropped.npy')


        elif upsample_factor==2:
            example_deblurred_im = np.load('dTV/Ptycho_XRF_project/Results/non_blind_deblurred_wheel_l1_up_2_non_cropped.npy')
            # example_deblurred_im_padded = np.ones((height, width))
            # example_deblurred_im_padded[kernel_width:-kernel_width, kernel_width:-kernel_width] = np.load('dTV/Ptycho_XRF_project/Results/non_blind_deblurred_wheel_l1.npy')

    elif dataset=='dataset_2':
        if upsample_factor==1:
            print("yes")
            im = Image.fromarray(ptycho)
            ptycho_image_resized = np.array(im.resize((width, height)))

        sinfo = ptycho_image_resized

    # checkerboard_2 = compare_images(example_deblurred_im, sinfo, method='checkerboard', n_tiles=(16, 16))
    #
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # fig.suptitle('Checking alignment of guide image')
    # ax1.imshow(example_deblurred_im, cmap=plt.cm.gray)
    # ax2.imshow(sinfo, cmap=plt.cm.gray)
    # ax3.imshow(checkerboard_2, cmap=plt.cm.gray)

    # if upsample_factor != 1:  # TODO This is a hack - the sampling operator used below must take the transpose
    #     sinfo = sinfo.T

    #alpha = 0.1
    #alpha = 0.01
    alpha = 0.01
    eta = 0.1
    #eta = 0.01
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

# let's see what the convolution of the ptycho image looks like
# convolved_ptycho = conv(image_space.element(ptycho_downsampled))
# convolved_ptycho.show()
# data.show()

if upsample_factor == 1:
    forward_op = conv
    sampling_points = get_central_sampling_points(data.shape, image_space.shape)
    emb = Embedding(data_space, image_space, sampling_points=sampling_points, adjoint=None)
    forward_op = emb.adjoint * conv

else:
    vert_ind = np.sort(list(1 + kernel_width+int(upsample_factor)*np.arange(data_height))*int(data_width))
    horiz_ind = list(1 + kernel_width + int(upsample_factor) * np.arange(data_width)) * int(data_height)
    sampling_points = [horiz_ind, vert_ind]
    emb = Embedding(data_space, image_space, sampling_points=sampling_points, adjoint=None)
    forward_op = emb.adjoint*conv

op_norm = 1.1 * odl.power_method_opnorm(forward_op)
tau = 1.0 / op_norm  # Step size for the primal variable
sigma = 1.0 / op_norm  # Step size for the dual variable

# trying to do non-blind deconvolution

x = image_space.zero()

niter = 200
st = 10
function_value = datafit*forward_op + reg_image
cb = (odl.solvers.CallbackPrintIteration(fmt='iter:{:4d}', step=st, end=', ') &
      odl.solvers.CallbackPrintTiming(fmt='time: {:5.2f} s', cumulative=True, step=st, end=', ') &
      odl.solvers.CallbackShow(step=10) &
      odl.solvers.CallbackPrint(function_value, fmt='f(x)={0:.4g}', step=st) &
      odl.solvers.CallbackPrint(datafit*forward_op, fmt='datafit={0:.4g}', step=st) &
      odl.solvers.CallbackPrint(reg_image, fmt='reg={0:.4g}', step=st) &
      odl.solvers.CallbackShowConvergence(function_value))

odl.solvers.pdhg(x, reg_image, datafit, forward_op, niter=niter, tau=tau, sigma=sigma, callback=cb)

## plotting results

# if upsample_factor==1:
#     recon = x.asarray()
#
# else:
#     recon = x.asarray().T

recon = x.asarray()

#discr = np.abs(forward_op(x).asarray()[margin_width:-margin_width, margin_width:-margin_width] - XRF_image/np.amax(XRF_image))
#discr = np.abs(forward_op(x).asarray() - XRF_image/np.amax(XRF_image))

if data_fit=='l2sq':
    discr = np.abs(forward_op(x).asarray() - data.asarray())

# elif datafit=='KL':
#     discr =


fig, (ax3, ax4, ax5, ax6) = plt.subplots(1, 4, figsize=(20, 4))
fig.suptitle('Results using '+ str(reg_type) + " regulariser, " + str(data_fit) + " datafit")
#im_1 = ax1.imshow(sinfo.T, cmap=plt.cm.gray)
# im_2 = ax2.imshow(recon, cmap=plt.cm.gray)
# ax2.axis("off")
# ax2.set_title("Recon.")
im_3 = ax3.imshow(recon[kernel_width:-kernel_width, kernel_width:-kernel_width], cmap=plt.cm.gray)#, vmax=0.5)
ax3.set_title("Recon.")
im_4 = ax4.imshow(forward_op(x).asarray(), cmap=plt.cm.gray)
ax4.set_title("Blurred recon.")
im_5 = ax5.imshow(data.asarray(), cmap=plt.cm.gray)
ax5.axis("off")
ax5.set_title("Data")
im_6 = ax6.imshow(discr, cmap=plt.cm.gray)
ax6.axis("off")
ax6.set_title("Resid.")
divider_3 = make_axes_locatable(ax3)
divider_4 = make_axes_locatable(ax4)
divider_5 = make_axes_locatable(ax5)
divider_6 = make_axes_locatable(ax6)
cax3 = divider_3.append_axes("right", size="5%", pad=0.04)
cax4 = divider_4.append_axes("right", size="5%", pad=0.04)
cax5 = divider_5.append_axes("right", size="5%", pad=0.04)
cax6 = divider_6.append_axes("right", size="5%", pad=0.04)
plt.colorbar(im_3, cax=cax3)
plt.colorbar(im_4, cax=cax4)
plt.colorbar(im_5, cax=cax5)
plt.colorbar(im_6, cax=cax6)
plt.tight_layout(rect=[1, 1, 1, 1])


convolved_sinfo = forward_op(image_space.element(sinfo)).asarray()
convolved_sinfo_rescaled = (convolved_sinfo - np.amin(convolved_sinfo))/(np.amax(convolved_sinfo) - np.amin(convolved_sinfo))
checkerboard_3 = compare_images(convolved_sinfo_rescaled, data.asarray(), method='checkerboard', n_tiles=(20, 20))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(convolved_sinfo_rescaled, cmap=plt.cm.gray)
ax2.imshow(data.asarray(), cmap=plt.cm.gray)
ax3.imshow(checkerboard_3, cmap=plt.cm.gray)

###
# im = Image.fromarray(ptycho)
# # new_image = np.array(im.resize(size, PIL.Image.BICUBIC))
# ptycho_resized = np.array(im.resize((width, height)))
# ptycho_resized -= np.amin(ptycho_resized)
#
# ptycho_resized_odl = image_space.element(ptycho_resized)
# conv(ptycho_resized_odl).show()

# ptycho_rescaled = (sinfo - np.amin(sinfo))/(np.amax(sinfo) - np.amin(sinfo))
#
# forward_op(sinfo).show()
# data.show()
# np.abs(forward_op(sinfo.T) - data).show()
#
# forward_op(sinfo.T).show()
# data.show()
# np.abs(forward_op(sinfo.T) - data).show()

