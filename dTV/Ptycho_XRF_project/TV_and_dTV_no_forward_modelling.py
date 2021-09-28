# In this script, I initialise the reconstruction at the rescaled, registered pytchographic recon,
# and perform updates only only the kernel. The aim is to see how well the problem is modelled by
# as a de-blurring problem. We're assuming that the ptychographic reconstruction can be considered a
# reasonable ground-truth for XRF in this case.

import odl
import numpy as np
from PIL import Image
from processing import *
import dTV.myAlgorithms as algs
import dTV.myFunctionals as fctls

# grabbing data
XRF_image = np.load('dTV/CT_data/Ptycho_XRF_07042021/XRF_W_La.npy')
ptycho = np.load('dTV/CT_data/Ptycho_XRF_07042021/Ptycho.npy')

# modifying field-of-view of ptycho and rescaling pixel intensity

ptycho_window = ptycho[372:-351, 369:-354]

im = Image.fromarray(ptycho_window)
# new_image = np.array(im.resize(size, PIL.Image.BICUBIC))
ptycho_resized = np.array(im.resize(XRF_image.shape))
ptycho_resized -= np.amin(ptycho_resized)

plt.imshow(ptycho_resized, cmap=plt.cm.gray)
plt.colorbar()

height, width = XRF_image.shape
image_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
                                          shape=[height, width], dtype='float')

# TV
model = VariationalRegClass('STEM', 'TV')
recons = model.regularised_recons_from_subsampled_data(XRF_image/np.amax(XRF_image), 0.01, niter=500)

plt.figure()
plt.imshow(XRF_image, cmap=plt.cm.gray)

plt.figure()
plt.imshow(recons[0], cmap=plt.cm.gray)

# dTV
#alpha = 1.0
alpha = 0.001
eta = 0.01
gamma = 0.9995
strong_cvx = 1e-5
niter_prox = 20
niter = 150

prox_options = {}
prox_options['name'] = 'FGP'
prox_options['warmstart'] = True
prox_options['p'] = None
prox_options['tol'] = None
prox_options['niter'] = niter_prox

data_odl = image_space.element(XRF_image/np.amax(XRF_image))
sinfo = ptycho_resized
Yaff = odl.tensor_space(6)
X = odl.ProductSpace(image_space, Yaff)

forward_op = odl.IdentityOperator(image_space)

reg_affine = odl.solvers.ZeroFunctional(Yaff)
x0 = X.element([forward_op.adjoint(data_odl), X[1].zero()])

f = fctls.DataFitL2Disp(X, data_odl, forward_op)

cb = (odl.solvers.CallbackPrintIteration(end=', ') &
      odl.solvers.CallbackPrintTiming(cumulative=False, end=', ') &
      odl.solvers.CallbackPrintTiming(fmt='total={:.3f}s', cumulative=True) &
      odl.solvers.CallbackShow(step=5))

reg_im = fctls.directionalTotalVariationNonnegative(image_space, alpha=alpha, sinfo=sinfo,
                                                                                gamma=gamma, eta=eta, NonNeg=True,
                                                    strong_convexity=strong_cvx, prox_options=prox_options)

g = odl.solvers.SeparableSum(reg_im, reg_affine)

cb = (odl.solvers.CallbackPrintIteration(end=', ') &
      odl.solvers.CallbackPrintTiming(cumulative=False, end=', ') &
      odl.solvers.CallbackPrintTiming(fmt='total={:.3f}s', cumulative=True) &
      odl.solvers.CallbackShow(step=5))

L = [1, 1e+2]
ud_vars = [0, 1]

# %%
#palm = algs.PALM(f, g, ud_vars=ud_vars, x=x0.copy(), callback=None, L=L)
palm = algs.PALM(f, g, ud_vars=ud_vars, x=x0.copy(), callback=cb, L=L)
palm.run(niter)

palm.x.show()

data_odl.show()
image_space.element(sinfo).show()

np.abs((palm.x[0] - data_odl)).show()
data_odl.show()
