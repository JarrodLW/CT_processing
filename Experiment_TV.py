import h5py
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from processing import *
import dTV.myFunctionals as fctls
import dTV.myAlgorithms as algs
import json

filename_XRF = 'Data/Experiment1_XRF.hdf5'
f1 = h5py.File(filename_XRF, 'r+')

filename_XRD = 'Data/Experiment1_XRD.hdf5'
f2 = h5py.File(filename_XRD, 'r+')

sino_Co = np.array(f1['sino_Co'])
sino_Co_1 = sino_Co[:, :, 0]

data_XRD = np.array(f2['sino_XRD'])

## pre-processing of XRD data: cumulative sum of hits within specified freq range
data_XRD_0 = data_XRD[:, :, :, 0]
# summing over pixels
plt.figure()
plt.hist(np.sum(data_XRD_0, axis=(0, 1)), range=(0, 4000), bins=200)
plt.title("Spectrum obtained by summing over pixels")
plt.show()

# we select the range 450-500, corresponding roughly to the second spectral peak
filter = np.zeros(data_XRD.shape[2])
filter[450:500] = 1
sino_0_XRD = np.dot(data_XRD[:, :, :, 0], filter)  # put processed sinogram here!

# rescaling the data to give it unit L2-weight
sino_Co_1_normalised = sino_Co_1/np.sqrt(np.sum(np.square(sino_Co_1)))
sino_0_XRD_normalised = sino_0_XRD/np.sqrt(np.sum(np.square(sino_0_XRD)))

## TV-regularised reconstructions, XRF and XRD
# model parameter
reg_param = 10**(-5)

# metadata
model = VariationalRegClass('CT', 'TV')
a_offset = -np.pi
a_range = 2*np.pi
d_offset = 0
d_width = 40
niter = 100
recon_dim = sino_Co_1.shape[0]

# running the algorithm
recons_XRF_TV = model.regularised_recons_from_subsampled_data(sino_Co_1_normalised.T, reg_param, recon_dims=(recon_dim, recon_dim),
                                                              niter=niter, a_offset=a_offset, enforce_positivity=True,
                                                              a_range=a_range, d_offset=d_offset, d_width=d_width)[0]

recons_XRD_TV = model.regularised_recons_from_subsampled_data(sino_0_XRD_normalised.T, reg_param, recon_dims=(recon_dim, recon_dim),
                                                              niter=niter, a_offset=a_offset, enforce_positivity=True,
                                                              a_range=a_range, d_offset=d_offset, d_width=d_width)[0]

plt.figure()
plt.plot(recons_XRF_TV, cmap=plt.cm.gray)
plt.title("XRF reconstruction using TV regulariser")

plt.figure()
plt.plot(recons_XRD_TV, cmap=plt.cm.gray)
plt.title("XRD reconstruction using TV regulariser")


## TV-regularised XRD reconstructions, masking out the spikes
model = VariationalRegClass('CT', 'TV')
a_offset = -np.pi
a_range = 2 * np.pi
d_offset = 0
d_width=40
recon_dim = sino_Co_1.shape[0]
reg_param = 10**(-2.5)

spike_subsampling_arr = (sino_0_XRD < 0.18)*np.ones(sino_0_XRD.shape) # this is ad hoc!

# running the algorithm
recons_XRD_TV = model.regularised_recons_from_subsampled_data(sino_0_XRD_normalised.T, reg_param, recon_dims=(width, width),
                                                              subsampling_arr=spike_subsampling_arr.T,
                                                              niter=1000, a_offset=a_offset,
                                                              enforce_positivity=True,
                                                              a_range=a_range, d_offset=d_offset, d_width=d_width)[0]

plt.figure()
plt.plot(recons_XRD_TV, cmap=plt.cm.gray)
plt.title("XRD reconstruction using TV regulariser and spike mask")
