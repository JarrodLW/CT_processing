import odl
import numpy as np
from dTV.Ptycho_XRF_project.misc import *
import matplotlib.pyplot as plt

height = width = 128
image_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
                                          shape=[height, width], dtype='float')
phantom = odl.phantom.transmission.shepp_logan(image_space, modified=True)

kernel_height = 5
kernel_width = 5
kernel_space = odl.uniform_discr(min_pt=[-1., -1.], max_pt=[1., 1.],
                                          shape=[kernel_height, kernel_width], dtype='float')

kernel = kernel_space.element(np.ones((kernel_height, kernel_width)))/(kernel_height*kernel_width)
conv = Convolution(image_space, kernel)

phantom = odl.phantom.transmission.shepp_logan(image_space, modified=True)
phantom_blurred = conv(phantom)

# the datafit, with convolution
lin_op = odl.IdentityOperator(image_space)
domain = odl.space.pspace.ProductSpace(image_space, kernel_space)
datafit = DataFitL2LinearPlusConv(domain, lin_op, phantom_blurred)

# testing gradient of datafit with respect to image
base_point = domain.element([0.5*phantom, kernel])
pert_1 = np.zeros((height, width))
pert_1[::2, :] = 1/64
pert = domain.element([pert_1, np.zeros((kernel_height, kernel_width))])
pert_scales = np.logspace(-10, -2, num=50)

diffs_0 = []
diffs_1 = []

for pert_scale in pert_scales:

    diff_0 = np.abs(datafit(base_point + pert_scale*pert) - datafit(base_point))
    diff_1 = np.abs(datafit(base_point + pert_scale*pert) - datafit(base_point)
            - pert_scale*datafit.gradient[0](base_point).inner(image_space.element(pert_1)))

    diffs_0.append(diff_0)
    diffs_1.append(diff_1)

plt.plot(np.log10(pert_scales), np.log10(np.asarray(diffs_0)), label='zeroth order approx')
plt.plot(np.log10(pert_scales), np.log10(np.asarray(diffs_1)), label='first order approx')
plt.legend()

# testing gradient of datafit with respect to kernel
base_point = domain.element([0.5*phantom, kernel])
pert_1 = np.zeros((kernel_height, kernel_width))
pert_1[::2, :] = 1
pert = domain.element([np.zeros((height, width)), pert_1])
pert_scales = np.logspace(-10, -2, num=50)

diffs_0 = []
diffs_1 = []

for pert_scale in pert_scales:

    diff_0 = np.abs(datafit(base_point + pert_scale*pert) - datafit(base_point))
    diff_1 = np.abs(datafit(base_point + pert_scale*pert) - datafit(base_point)
            - pert_scale*datafit.gradient[1](base_point).inner(kernel_space.element(pert_1)))

    diffs_0.append(diff_0)
    diffs_1.append(diff_1)

plt.plot(np.log10(pert_scales), np.log10(np.asarray(diffs_0)), label='zeroth order approx')
plt.plot(np.log10(pert_scales), np.log10(np.asarray(diffs_1)), label='first order approx')
plt.legend()
