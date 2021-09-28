import h5py
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from mpl_toolkits.axes_grid1 import make_axes_locatable

file_ptycho = 'dTV/CT_data/Ptycho_XRF_27042021/scan_104797SI14G00_phase.nxs'
file_XRF = 'dTV/CT_data/Ptycho_XRF_27042021/i14-104797-xrf_windows-xsp3_addetector.nxs'
file_XRF_GT = 'dTV/CT_data/Ptycho_XRF_27042021/i14-104795-xrf_windows-xsp3_addetector.nxs'
phase_contrast = 'dTV/CT_data/Ptycho_XRF_27042021/dpc_i14-104797pdq2.nxs'

f_ptycho = h5py.File(file_ptycho, 'r+')
recon_ptycho = f_ptycho['entry']['phase_SI14G00']['data'][:, :, 0, 0]

f_XRF = h5py.File(file_XRF, 'r+')
recons_XRF = np.zeros((13, 80, 68))
recons_XRF[0] = f_XRF['processed']['auxiliary']['0-XRF Elemental Maps from ROIs']['Au-La']['data']
recons_XRF[1] = f_XRF['processed']['auxiliary']['0-XRF Elemental Maps from ROIs']['Ca-Ka']['data']
recons_XRF[2] = f_XRF['processed']['auxiliary']['0-XRF Elemental Maps from ROIs']['Co-Ka']['data']
recons_XRF[3] = f_XRF['processed']['auxiliary']['0-XRF Elemental Maps from ROIs']['Cu-Ka']['data']
recons_XRF[4] = f_XRF['processed']['auxiliary']['0-XRF Elemental Maps from ROIs']['Fe-Ka']['data']
recons_XRF[5] = f_XRF['processed']['auxiliary']['0-XRF Elemental Maps from ROIs']['Hg-La']['data']
recons_XRF[6] = f_XRF['processed']['auxiliary']['0-XRF Elemental Maps from ROIs']['K-Ka']['data']
recons_XRF[7] = f_XRF['processed']['auxiliary']['0-XRF Elemental Maps from ROIs']['Mn-Ka']['data']
recons_XRF[8] = f_XRF['processed']['auxiliary']['0-XRF Elemental Maps from ROIs']['P-Ka']['data']
recons_XRF[9] = f_XRF['processed']['auxiliary']['0-XRF Elemental Maps from ROIs']['S-Ka']['data']
recons_XRF[10] = f_XRF['processed']['auxiliary']['0-XRF Elemental Maps from ROIs']['Se-Ka']['data']
recons_XRF[11] = f_XRF['processed']['auxiliary']['0-XRF Elemental Maps from ROIs']['V-Ka']['data']
recons_XRF[12] = f_XRF['processed']['auxiliary']['0-XRF Elemental Maps from ROIs']['Zn-Ka']['data']

fig, axs = plt.subplots(3, 5, figsize=(10, 6))
for k, ax in enumerate(axs.flat):

    if k==13 or k==14:
        recon = np.ones(recons_XRF[0].shape)

    else:
        recon = recons_XRF[k]

    pcm = ax.imshow(recon, vmin=np.amin(recons_XRF), vmax=0.3*np.amax(recons_XRF), cmap=plt.cm.gray)
    ax.axis("off")

    if k<=12:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.04)
        plt.colorbar(pcm, cax=cax)

plt.tight_layout()

plt.imshow(np.exp(recons_XRF[10]/10), cmap=plt.cm.gray)

plt.figure()
plt.imshow(recons_XRF[1] + recons_XRF[3] + recons_XRF[4] + recons_XRF[9] + recons_XRF[12], cmap=plt.cm.gray)

f_XRF_GT = h5py.File(file_XRF_GT, 'r+')
recon_XRF_Ca_Ka_GT = f_XRF_GT['processed']['auxiliary']['0-XRF Elemental Maps from ROIs']['Ca-Ka']['data']


plt.figure()
plt.imshow(recon_ptycho, cmap=plt.cm.gray)

plt.figure()
plt.imshow(recon_XRF_Ca_Ka_GT, cmap=plt.cm.gray)

np.save('dTV/CT_data/Ptycho_XRF_27042021/Ptycho.npy', recon_ptycho)
np.save('dTV/CT_data/Ptycho_XRF_27042021/XRF_Ca_Ka.npy', recon_XRF_Ca_Ka)
np.save('dTV/CT_data/Ptycho_XRF_27042021/XRF_Ca_Ka_high_res.npy', recon_XRF_Ca_Ka_GT)


# phase contrast image
f_phase_contrast = h5py.File(phase_contrast, 'r+')

image = f_phase_contrast['entry002']['Phase']['data']

plt.figure()
plt.imshow(image, cmap=plt.cm.gray)

np.save('dTV/CT_data/Ptycho_XRF_27042021/phase_contrast.npy', image)

ptycho_im = np.load('dTV/CT_data/Ptycho_XRF_27042021/Ptycho.npy')
ptycho_im_downsized = resize(ptycho_im, (80, 68))

plt.figure()
plt.imshow(ptycho_im_downsized, cmap=plt.cm.gray)
