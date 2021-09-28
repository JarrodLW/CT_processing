import h5py
import matplotlib.pyplot as plt
import numpy as np

file_ptycho = 'dTV/CT_data/Ptycho_XRF_07042021/59719_20200109-115139.hdf'
file_XRF = 'dTV/CT_data/Ptycho_XRF_07042021/i14-59719-xrf_windows-xsp3_addetector2.nxs'

f_ptycho = h5py.File(file_ptycho, 'r+')

recon_ptycho = f_ptycho['entry_1']['process_1']['output_1']['object_phase'][0, 0, 0, 0, 0, :, :]

probe_mod = f_ptycho['entry_1']['process_1']['output_1']['probe_modulus'][0, 0, 0, 0, 0, :, :]

plt.figure()
plt.imshow(recon_ptycho, cmap=plt.cm.gray)

f_XRF = h5py.File(file_XRF, 'r+')
recon_XRF_Fe_Ka = f_XRF['processed']['auxiliary']['0-XRF Elemental Maps from ROIs']['Fe-Ka']['data']
recon_XRF_W_La = f_XRF['processed']['auxiliary']['0-XRF Elemental Maps from ROIs']['W-La']['data']

np.save('dTV/CT_data/Ptycho_XRF_07042021/Ptycho.npy', recon_ptycho)
np.save('dTV/CT_data/Ptycho_XRF_07042021/XRF_Fe_Ka.npy', recon_XRF_Fe_Ka)
np.save('dTV/CT_data/Ptycho_XRF_07042021/XRF_W_La.npy', recon_XRF_W_La)
np.save('dTV/CT_data/Ptycho_XRF_07042021/probe_modulus.npy', probe_mod)

plt.figure()
plt.imshow(recon_XRF_W_La, cmap=plt.cm.gray)
plt.colorbar()

plt.figure()
plt.imshow(recon_XRF_Fe_Ka, cmap=plt.cm.gray)



