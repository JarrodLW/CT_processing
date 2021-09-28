# created 27/04/2021. Here I want to perform TV and dTV-upsampled reconstructions of the 27/04/2021 CT dataset.
# The forward operator is just a subsampling operator

from PIL import Image
import numpy as np
import scipy as sp
import odl
import matplotlib.pyplot as plt
from dTV.Ptycho_XRF_project.misc import TotalVariationNonNegative, Embedding, EmbeddingAdjoint
import dTV.myFunctionals as fctls
from mpl_toolkits.axes_grid1 import make_axes_locatable

ptycho_image = np.load('dTV/CT_data/Ptycho_XRF_27042021/Ptycho.npy')
XRF_image = np.load('dTV/CT_data/Ptycho_XRF_27042021/XRF_Ca_Ka.npy')

fit_type = 'L2sq'
#fit_type = 'KL'
reg_type = 'TVNN'
#reg_param = 0.018
#reg_params = [0.01, 0.025, 0.035, 0.05, 0.1]
reg_params = [0.1]
upsample_factor = 1.  # upsampling not quite working yet. Compare with Nonblind_deconvolution_XRF_with_PDHG script.

## (d)TV denoising

height = int(upsample_factor*XRF_image.shape[0])
width = int(upsample_factor*XRF_image.shape[1])
image_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
                                          shape=[height, width], dtype='float')

im = Image.fromarray(ptycho_image)
ptycho_image_resized = np.array(im.resize((width, height)))

data_height, data_width = XRF_image.shape
data_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
                                          shape=[data_height, data_width], dtype='float')

# grabbing data
XRF_image_odl = data_space.element(XRF_image)
ptycho_image_odl = image_space.element(ptycho_image_resized)

# XRF_image_odl.show()
# ptycho_image_odl.show()

# checking registration - not sure yet how to do this
# XRF_image_odl_normalised = XRF_image_odl/np.amax(XRF_image_odl)
# ptycho_image_odl_normalised = (ptycho_image_odl - np.amin(ptycho_image_odl))/np.amax(ptycho_image_odl - np.amin(ptycho_image_odl))

# XRF_image_odl_normalised.show()
# ptycho_image_odl_normalised.show()
# (XRF_image_odl_normalised - ptycho_image_odl_normalised).show()

# forward op
#forward_op = odl.IdentityOperator(image_space)
if upsample_factor == 1:
    forward_op = odl.IdentityOperator(image_space)

else:
    # vert_ind = np.sort(list(int(upsample_factor)*np.arange(data_height))*int(data_width))
    # horiz_ind = list(int(upsample_factor)*np.arange(data_width))*int(data_height)
    vert_ind = np.sort(list(int(upsample_factor) * np.arange(data_height)) * int(data_width))
    horiz_ind = list(int(upsample_factor) * np.arange(data_width)) * int(data_height)
    sampling_points = [horiz_ind, vert_ind]
    emb = Embedding(data_space, image_space, sampling_points=sampling_points, adjoint=None)
    forward_op = emb.adjoint
    #forward_op = EmbeddingAdjoint(image_space, data_space, sampling_points, emb)

data = XRF_image_odl
#data = odl.phantom.noise.white_noise(image_space)**2 # a sanity check that we're not massively overfitting

if fit_type=='L2sq':
    datafit = odl.solvers.L2NormSquared(data_space).translated(data)

elif fit_type=='L1':
    datafit = odl.solvers.L1Norm(data_space).translated(data)

elif fit_type=='KL':
    data = np.round(data)
    datafit = odl.solvers.functional.default_functionals.KullbackLeibler(data_space, prior=data)

recons = []
for reg_param in reg_params:
    if reg_type=='TVNN':
        reg = reg_param*TotalVariationNonNegative(image_space)

    elif reg_type=='dTVNN': #TODO re-implement this as PDHG
        #eta = 0.01
        eta = 0.01
        gamma = 0.9995
        #gamma = 0.995
        niter_prox = 20
        prox_options = {}
        prox_options['name'] = 'FGP'
        prox_options['warmstart'] = True
        prox_options['p'] = None
        prox_options['tol'] = None
        prox_options['niter'] = niter_prox
        reg = fctls.directionalTotalVariationNonnegative(image_space, alpha=reg_param, sinfo=ptycho_image_odl,
                                                            gamma=gamma, eta=eta, NonNeg=True,
                                                            prox_options=prox_options)

    op_norm = 1.1 * odl.power_method_opnorm(forward_op)
    tau = 1.0 / op_norm  # Step size for the primal variable
    sigma = 1.0 / op_norm  # Step size for the dual variable

    x = image_space.zero()
    niter = 300

    st = 10
    function_value = datafit*forward_op + reg
    cb = (odl.solvers.CallbackPrintIteration(fmt='iter:{:4d}', step=st, end=', ') &
          odl.solvers.CallbackPrintTiming(fmt='time: {:5.2f} s', cumulative=True, step=st, end=', ') &
          odl.solvers.CallbackShow(step=10) &
          odl.solvers.CallbackPrint(function_value, fmt='f(x)={0:.4g}', step=st) &
          odl.solvers.CallbackPrint(datafit * forward_op, fmt='datafit={0:.4g}', step=st) &
          odl.solvers.CallbackPrint(reg, fmt='reg={0:.4g}', step=st) &
          odl.solvers.CallbackShowConvergence(function_value))

    odl.solvers.pdhg(x, reg, datafit, forward_op, niter=niter, tau=tau, sigma=sigma, callback=cb)

    recon = x.asarray()

    recons.append(recon)

fig, axarr = plt.subplots(len(reg_params), 3, figsize=(4, 2*len(reg_params)))
vmax = max(np.amax(np.asarray(recons)), np.amax(data.asarray()))

for i in range(len(reg_params)):

    recon = recons[i]

    if datafit=='l2sq':
        res = np.abs(recon - data.asarray())

    elif datafit=='KL':
        from scipy.stats import poisson

        rv = poisson(data.asarray())
        res = rv.pmf(x.asarray()//1)

    im_1 = axarr[i, 0].imshow(recon, cmap=plt.cm.gray)#, vmax=vmax)
    axarr[i, 0].axis("off")
    im_2 = axarr[i, 1].imshow(data.asarray(), cmap=plt.cm.gray)#, vmax=vmax)
    axarr[i, 1].axis("off")
    im_3 = axarr[i, 2].imshow(res, cmap=plt.cm.gray)#, vmax=vmax)
    axarr[i, 2].axis("off")

    divider_1 = make_axes_locatable(axarr[i, 0])
    divider_2 = make_axes_locatable(axarr[i, 1])
    divider_3 = make_axes_locatable(axarr[i, 2])
    cax1 = divider_1.append_axes("right", size="5%", pad=0.04)
    cax2 = divider_2.append_axes("right", size="5%", pad=0.04)
    cax3 = divider_3.append_axes("right", size="5%", pad=0.04)
    plt.colorbar(im_1, cax=cax1)
    plt.colorbar(im_2, cax=cax2)
    plt.colorbar(im_3, cax=cax3)

axarr[0, 0].set_title("Recon.")
axarr[0, 1].set_title("Data")
axarr[0, 2].set_title("Resid.")

plt.tight_layout(rect=[3, 3])


#
# from dTV.Ptycho_XRF_project.misc import Embedding
# from dTV.Ptycho_XRF_project.misc import get_central_sampling_points
#
# X = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
#                                           shape=[7, 5], dtype='float')
# Y = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
#                                           shape=[28, 20], dtype='float')
#
# sampling_points = [np.sort([0, 4, 8, 12, 16, 20, 24]*5), [0, 4, 8, 12, 16]*7]
#
# emb = Embedding(X, Y, sampling_points=sampling_points, adjoint=None)
#
# A = np.reshape(np.arange(35), (7, 5))
# B = emb(A)
#
# plt.figure()
# plt.imshow(A, cmap=plt.cm.gray)
#
# plt.figure()
# plt.imshow(B, cmap=plt.cm.gray)
#
# plt.figure()
# plt.imshow(emb.adjoint(B), cmap=plt.cm.gray)

# from scipy.stats import poisson
#
# rv = poisson(data.asarray())
# rv.pmf(x.asarray()//1)
#
# plt.figure()
# plt.imshow(rv.pmf(data.asarray()//1), cmap=plt.cm.gray)
# plt.colorbar()
#
# plt.figure()
# plt.imshow(x.asarray(), cmap=plt.cm.gray)

# residual
# with np.errstate(invalid='ignore', divide='ignore'):
#     xlogy = sp.special.xlogy(data.asarray(), data.asarray() / x)
#     res = (x - data.asarray() + xlogy)
