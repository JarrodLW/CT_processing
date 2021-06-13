import h5py
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from processing import *
import dTV.myFunctionals as fctls
import dTV.myAlgorithms as algs
import json

guide_image = np.load('Data/pre_registered_XRF_image.npy')

filename_XRD = 'Data/Experiment1_XRD.hdf5'
f2 = h5py.File(filename_XRD, 'r+')

data_XRD = np.array(f2['sino_XRD'])

## pre-processing of XRD data: cumulative sum of hits within specified freq range - same as in TV experiment script
data_XRD_0 = data_XRD[:, :, :, 0]

# we select the range 450-500, corresponding roughly to the second spectral peak
filter = np.zeros(data_XRD.shape[2])
filter[450:500] = 1
sino_0_XRD = np.dot(data_XRD[:, :, :, 0], filter)  # put processed sinogram here!

# rescaling the data to give it unit L2-weight
sino_0_XRD_normalised = sino_0_XRD/np.sqrt(np.sum(np.square(sino_0_XRD)))

## guided processing

# model parameters
gamma = 0.995
alpha = 10**(-4)
eta = 0.001
strong_cvx = 1e-5

# metadata
niter_prox = 20
niter = 50
a_offset = -np.pi
a_range = 2 * np.pi
d_offset = 0
d_width = 40
recon_dim = sino_0_XRD.shape[0]

Yaff = odl.tensor_space(6)

data = sino_0_XRD_normalised.T
height, width = data.shape

image_space = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20], shape=[recon_dim, recon_dim], dtype='float')
# Make a parallel beam geometry with flat detector
angle_partition = odl.uniform_partition(a_offset, a_offset+a_range, height)
# Detector: uniformly sampled
detector_partition = odl.uniform_partition(d_offset-d_width/2, d_offset+d_width/2, width)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

# creating a mask to mask out spikes in sinogram - ad hoc!
subsampling_arr = (sino_0_XRD < 0.18) * np.ones(sino_0_XRD.shape)
masked_data = (subsampling_arr.T)*data

# Create the forward operator
forward_op_CT = odl.tomo.RayTransform(image_space, geometry, impl='skimage')
subsampled_forward_op = forward_op_CT.range.element(subsampling_arr.T)*forward_op_CT

data_odl = forward_op_CT.range.element(masked_data)
sinfo = image_space.element(guide_image.T[:, ::-1])

# space of optimised variables
X = odl.ProductSpace(image_space, Yaff)

# Set some parameters and the general TV prox options
prox_options = {}
prox_options['name'] = 'FGP'
prox_options['warmstart'] = True
prox_options['p'] = None
prox_options['tol'] = None
prox_options['niter'] = niter_prox

reg_affine = odl.solvers.ZeroFunctional(Yaff)
x0 = X.zero()

f = fctls.DataFitL2Disp(X, data_odl, subsampled_forward_op)

reg_im = fctls.directionalTotalVariationNonnegative(image_space, alpha=alpha, sinfo=sinfo,
                                                    gamma=gamma, eta=eta, NonNeg=True, strong_convexity=strong_cvx,
                                                    prox_options=prox_options)

g = odl.solvers.SeparableSum(reg_im, reg_affine)

cb = (odl.solvers.CallbackPrintIteration(end=', ') &
      odl.solvers.CallbackPrintTiming(cumulative=False, end=', ') &
      odl.solvers.CallbackPrintTiming(fmt='total={:.3f}s', cumulative=True) &
      odl.solvers.CallbackShow()
      )

L = [1, 1e+2]
ud_vars = [0]  # Change this to [0, 1] to turn on simultaneous registration

# %%
palm = algs.PALM(f, g, ud_vars=ud_vars, x=x0.copy(), callback=None, L=L)
palm.run(niter)

recon = palm.x[0].asarray()
fidelity = f(palm.x)

plt.figure()
plt.imshow(recon, cmap=plt.cm.gray)


