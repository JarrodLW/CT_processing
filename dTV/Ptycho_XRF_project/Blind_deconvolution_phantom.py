import odl
import numpy as np
from dTV.Ptycho_XRF_project.misc import Convolution, TotalVariationNonNegative, DataFitL2LinearPlusConv, DataFitL2LinearPlusConvViaFFT, PALM
from odl.solvers import L2NormSquared as odl_l2sq
from scipy.ndimage import gaussian_filter

height = width = 128
image_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
                                          shape=[height, width], dtype='float')
phantom = odl.phantom.transmission.shepp_logan(image_space, modified=True)

kernel_height = 5
kernel_width = 5
kernel_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
                                          shape=[kernel_height, kernel_width], dtype='float')

kernel = kernel_space.element(np.random.normal(loc=0.0, scale=0.5, size=(kernel_height, kernel_width))**2)
kernel /= np.sum(kernel)
conv = Convolution(image_space, kernel)
phantom_blurred = conv(phantom)

kernel.show()

# phantom.show()
# phantom_blurred.show()
# conv.adjoint(phantom_blurred).show()

# datafit(datafit.domain.element([phantom, kernel]))
# odl.solvers.L2NormSquared(image_space)(conv(phantom) - phantom_blurred)

# regularisers
lambda_image = 0.0000001
TV_image = TotalVariationNonNegative(image_space, alpha=lambda_image)
kernel_reg = odl.solvers.IndicatorSimplex(kernel_space)

# the datafit, with convolution
lin_op = odl.IdentityOperator(image_space)
domain = odl.space.pspace.ProductSpace(image_space, kernel_space)
#datafit = DataFitL2LinearPlusConv(domain, lin_op, phantom_blurred)
datafit = DataFitL2LinearPlusConvViaFFT(domain, lin_op, phantom_blurred)


Reg = odl.solvers.SeparableSum(TV_image, kernel_reg)

#x = domain.element([7*phantom_blurred, kernel])
#x = domain.element([7*phantom_blurred, np.zeros((kernel_height, kernel_width))])
#x = domain.element([1.4*phantom_blurred, np.zeros((kernel_height, kernel_width))])
x = domain.element([phantom_blurred/np.amax(phantom_blurred), np.zeros((kernel_height, kernel_width))])
#x = domain.element([phantom_blurred, np.zeros((kernel_height, kernel_width))])
st = 1

L = [1, 1e2]
#L = [1, 1e3]
L = [1e2, 1e2]

ud_vars = [0, 1]

function_value = datafit + Reg
cb = (odl.solvers.CallbackPrintIteration(fmt='iter:{:4d}', step=st, end=', ') &
      odl.solvers.CallbackPrintTiming(fmt='time: {:5.2f} s', cumulative=True, step=st, end=', ') &
      odl.solvers.CallbackShow(step=5) &
      odl.solvers.CallbackPrint(function_value, fmt='f(x)={0:.4g}', step=st))

PALM(datafit, Reg, ud_vars=ud_vars, x=x, L=L, niter=500, callback=cb)

x.show()
phantom.show()
kernel.show()

#
# l2 = odl.solvers.L2Norm(image_space)
# phantom_normalised = phantom/l2(phantom)
# phantom_blurred_normalised = phantom_blurred/l2(phantom_blurred)
#
# phantom_normalised.show()
# phantom_blurred_normalised.show()
#
# TV_image(phantom_normalised)/TV_image(phantom_blurred_normalised)
#
# TV_image(phantom/np.amax(phantom))/TV_image(phantom_blurred/np.amax(phantom_blurred))



