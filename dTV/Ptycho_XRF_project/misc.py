# This is taken from the "blur" code that Matthias sent.

import os
import odl
import numpy as np
import matplotlib
from skimage.io import imsave
import matplotlib.pyplot as plt
from scipy.ndimage import convolve as sp_convolve
from scipy.signal import fftconvolve
from scipy.signal import convolve as signal_convolve
from odl.solvers import L2NormSquared as odl_l2sq
from odl.operator.operator import Operator
from time import time


__all__ = ('Convolution', 'ConvolutionEmbedding',
           'ConvolutionEmbeddingAdjoint',
           'get_central_sampling_points', 'Embedding', 'EmbeddingAdjoint',
           'ind_fun_circle2d', 'gaussian2d', 'total_variation',
           'TotalVariationNonNegative', 'fgp_dual', 'DataFitL2LinearPlusConv',
           'fbs', 'bregman_iteration', 'PALM', 'blind_bregman_iteration',
           'test_adjoint', 'save_image', 'save_ssim', 'dTV')


class Convolution(odl.Operator):

    def __init__(self, space, kernel, origin=None, adjoint=None):

        self.__kernel = kernel
        self.__origin = origin
        self.__adjoint = adjoint
        self.__boundary_condition = 'constant'
        self.__scale = kernel.space.domain.volume / len(kernel)

        super().__init__(space, space, linear=True)

    def _call(self, x, out):
        sp_convolve(x, self.kernel, output=out.asarray(),
                    mode=self.boundary_condition, origin=self.origin)

        out *= self.__scale

    @property
    def kernel(self):
        return self.__kernel

    @property
    def origin(self):
        if self.__origin is None:
            self.__origin = [0, ] * self.domain.ndim

        return self.__origin

    @property
    def boundary_condition(self):
        return self.__boundary_condition

    @property
    def adjoint(self):
        if self.__adjoint is None:
            if self.domain.ndim == 2:
                kernel = np.fliplr(np.flipud(self.kernel.copy().conj()))
                kernel = self.kernel.space.element(kernel)
            else:
                raise NotImplementedError('"adjoint_kernel" only defined for '
                                          '2d kernels')

            origin = [0, ] * self.domain.ndim
            for i in range(self.domain.ndim):
                if np.mod(self.kernel.shape[i], 2) == 0:
                    origin[i] = -1

            for i in range(self.domain.ndim):
                if kernel.shape[i] < 3 and origin[i] == -1:
                    NotImplementedError('Shifted origins are only implemented '
                                        'for kernels of size 3 or larger.')

            self.__adjoint = Convolution(self.domain, kernel, origin, self)

        return self.__adjoint

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r}, {!r}, {!r})'.format(
            self.__class__.__name__, self.domain, self.kernel, self.origin,
            self.boundary_condition)


class ConvolutionViaFFT(odl.Operator): #TODO this is totally hacked. Not using origin or boundary condition arguments

    def __init__(self, space, kernel, origin=None, adjoint=None):

        self.__kernel = kernel
        self.__origin = origin
        self.__adjoint = adjoint
        self.__boundary_condition = 'same'
        self.__scale = kernel.space.domain.volume / len(kernel)

        super().__init__(space, space, linear=True)

    def _call(self, x, out):

        #print(x.space)
        #print(fftconvolve(x, self.kernel, mode=self.boundary_condition).shape)

        out[:] = fftconvolve(x, self.kernel,  mode=self.boundary_condition)

        out *= self.__scale

    @property
    def kernel(self):
        return self.__kernel

    @property
    def origin(self):
        if self.__origin is None:
            self.__origin = [0, ] * self.domain.ndim

        return self.__origin

    @property
    def boundary_condition(self):
        return self.__boundary_condition

    @property
    def adjoint(self):
        if self.__adjoint is None:
            if self.domain.ndim == 2:
                kernel = np.fliplr(np.flipud(self.kernel.copy().conj()))
                kernel = self.kernel.space.element(kernel)
            else:
                raise NotImplementedError('"adjoint_kernel" only defined for '
                                          '2d kernels')

            origin = [0, ] * self.domain.ndim
            for i in range(self.domain.ndim):
                if np.mod(self.kernel.shape[i], 2) == 0:
                    origin[i] = -1

            for i in range(self.domain.ndim):
                if kernel.shape[i] < 3 and origin[i] == -1:
                    NotImplementedError('Shifted origins are only implemented '
                                        'for kernels of size 3 or larger.')

            self.__adjoint = ConvolutionViaFFT(self.domain, kernel, origin, self)

        return self.__adjoint

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r}, {!r}, {!r})'.format(
            self.__class__.__name__, self.domain, self.kernel, self.origin,
            self.boundary_condition)


class ConvolutionEmbedding(odl.Operator):

    def __init__(self, domain, range, kernel, origin=None, adjoint=None):

        self.__kernel = kernel
        self.__origin = origin
        self.__adjoint = adjoint
        self.__boundary_condition = 'constant'
        self.__scale = kernel.space.domain.volume / len(kernel)

        super().__init__(domain, range, linear=True)

    def _call(self, x, out):
        sp_convolve(self.kernel, x, output=out.asarray(),
                    mode=self.boundary_condition, origin=self.origin)

        out *= self.__scale

    @property
    def kernel(self):
        return self.__kernel

    @property
    def origin(self):
        if self.__origin is None:
            self.__origin = [0, ] * self.domain.ndim

        return self.__origin

    @property
    def boundary_condition(self):
        return self.__boundary_condition

    @property
    def adjoint(self):
        if self.__adjoint is None:
            origin = [0, ] * self.domain.ndim
            for i in range(self.domain.ndim):
                if np.mod(self.kernel.shape[i], 2) == 0:
                    origin[i] = -1

            for i in range(self.domain.ndim):
                if self.kernel.shape[i] < 3 and origin[i] == -1:
                    NotImplementedError('Shifted origins are only implemented '
                                        'for kernels of size 3 or larger.')

            self.__adjoint = ConvolutionEmbeddingAdjoint(self.range,
                                                         self.domain,
                                                         self.kernel,
                                                         origin, self)

        return self.__adjoint

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r}, {!r}, {!r})'.format(
            self.__class__.__name__, self.domain, self.kernel, self.origin,
            self.boundary_condition)


class ConvolutionEmbeddingAdjoint(odl.Operator):

    def __init__(self, domain, range, kernel, origin=None, adjoint=None):

        self.__kernel = kernel
        self.__origin = origin
        self.__adjoint = adjoint
        self.__boundary_condition = 'constant'
        self.__scale = kernel.space.domain.volume / len(kernel)

        super().__init__(domain, range, linear=True)

    def _call(self, x, out):
        if not self.domain.ndim == 2:
            raise NotImplementedError('adjoint only defined for 2d domains')

        out_a = out.asarray()
        x_a = x.asarray()
        k_a = self.kernel.asarray()

        n = x.shape
        s = out.shape[0] // 2, out.shape[1] // 2

        for i in range(out_a.shape[0]):

            if n[0] > 1:
                ix1, ix2 = max(i - s[0], 0), min(n[0] + i - s[0], n[0])
                ik1, ik2 = max(s[0] - i, 0), min(n[0] - i + s[0], n[0])
            else:
                ix1, ix2 = 0, 1
                ik1, ik2 = 0, 1

            for j in range(out_a.shape[1]):
                if n[1] > 1:
                    jx1, jx2 = max(j - s[1], 0), min(n[1] + j - s[1], n[1])
                    jk1, jk2 = max(s[1] - j, 0), min(n[1] - j + s[1], n[1])
                else:
                    jx1, jx2 = 0, 1
                    jk1, jk2 = 0, 1

                out_a[i, j] = np.sum(x_a[ix1:ix2, jx1:jx2] *
                                     k_a[ik1:ik2, jk1:jk2])

        out *= self.__scale

    @property
    def kernel(self):
        return self.__kernel

    @property
    def origin(self):
        if self.__origin is None:
            self.__origin = [0, ] * self.domain.ndim

        return self.__origin

    @property
    def boundary_condition(self):
        return self.__boundary_condition

    @property
    def adjoint(self):
        if self.__adjoint is None:
            NotImplementedError('Can only be called as an "adjoint" of '
                                '"ConvolutionEmbedding".')

        return self.__adjoint

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r}, {!r}, {!r})'.format(
            self.__class__.__name__, self.domain, self.kernel, self.origin,
            self.boundary_condition)


class ConvolutionEmbeddingViaFFT(odl.Operator): #TODO again, totally hacked

    def __init__(self, domain, range, kernel, origin=None, adjoint=None):

        self.__kernel = kernel
        self.__origin = origin
        self.__adjoint = adjoint
        self.__boundary_condition = 'same'
        self.__scale = kernel.space.domain.volume / len(kernel)

        super().__init__(domain, range, linear=True)

    def _call(self, x, out):

        out[:] = fftconvolve(x, self.kernel, mode=self.boundary_condition)

        out *= self.__scale

    @property
    def kernel(self):
        return self.__kernel

    @property
    def origin(self):
        if self.__origin is None:
            self.__origin = [0, ] * self.domain.ndim

        return self.__origin

    @property
    def boundary_condition(self):
        return self.__boundary_condition

    @property
    def adjoint(self):
        if self.__adjoint is None:
            origin = [0, ] * self.domain.ndim
            for i in range(self.domain.ndim):
                if np.mod(self.kernel.shape[i], 2) == 0:
                    origin[i] = -1

            for i in range(self.domain.ndim):
                if self.kernel.shape[i] < 3 and origin[i] == -1:
                    NotImplementedError('Shifted origins are only implemented '
                                        'for kernels of size 3 or larger.')

            self.__adjoint = ConvolutionEmbeddingAdjointViaFFT(self.range,
                                                         self.domain,
                                                         self.kernel,
                                                         origin, self)

        return self.__adjoint

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r}, {!r}, {!r})'.format(
            self.__class__.__name__, self.domain, self.kernel, self.origin,
            self.boundary_condition)


class ConvolutionEmbeddingAdjointViaFFT(odl.Operator): #TODO, totally hacked

    def __init__(self, domain, range, kernel, origin=None, adjoint=None):

        self.__kernel = kernel
        self.__origin = origin
        self.__adjoint = adjoint
        self.__boundary_condition = 'same'
        self.__scale = kernel.space.domain.volume / len(kernel)

        super().__init__(domain, range, linear=True)

    def _call(self, x, out):
        if not self.domain.ndim == 2:
            raise NotImplementedError('adjoint only defined for 2d domains')

        out_a = out.asarray()
        x_a = x.asarray()
        k_a = self.kernel.asarray()

        n = x.shape
        s = out.shape[0] // 2, out.shape[1] // 2

        for i in range(out_a.shape[0]):

            if n[0] > 1:
                ix1, ix2 = max(i - s[0], 0), min(n[0] + i - s[0], n[0])
                ik1, ik2 = max(s[0] - i, 0), min(n[0] - i + s[0], n[0])
            else:
                ix1, ix2 = 0, 1
                ik1, ik2 = 0, 1

            for j in range(out_a.shape[1]):
                if n[1] > 1:
                    jx1, jx2 = max(j - s[1], 0), min(n[1] + j - s[1], n[1])
                    jk1, jk2 = max(s[1] - j, 0), min(n[1] - j + s[1], n[1])
                else:
                    jx1, jx2 = 0, 1
                    jk1, jk2 = 0, 1

                out_a[i, j] = np.sum(x_a[ix1:ix2, jx1:jx2] *
                                     k_a[ik1:ik2, jk1:jk2])

        out *= self.__scale

    @property
    def kernel(self):
        return self.__kernel

    @property
    def origin(self):
        if self.__origin is None:
            self.__origin = [0, ] * self.domain.ndim

        return self.__origin

    @property
    def boundary_condition(self):
        return self.__boundary_condition

    @property
    def adjoint(self):
        if self.__adjoint is None:
            NotImplementedError('Can only be called as an "adjoint" of '
                                '"ConvolutionEmbedding".')

        return self.__adjoint

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r}, {!r}, {!r})'.format(
            self.__class__.__name__, self.domain, self.kernel, self.origin,
            self.boundary_condition)



def get_central_sampling_points(small_shape, large_shape):

    m = np.array(large_shape, dtype='int')
    n = np.array(small_shape, dtype='int')

    c1 = all(mi % 2 == 0 for mi in m)
    c2 = all(ni % 2 != 0 for ni in n)
    c3 = m[0] % 2 == 0 and m[1] % 2 != 0
    c4 = n[0] % 2 == 0 and n[1] % 2 != 0
    c5 = m[0] % 2 != 0 and m[1] % 2 == 0
    c6 = n[0] % 2 != 0 and n[1] % 2 == 0

    case1 = c1 and c4
    case2 = c1 and c2
    case3 = c5 and c2
    case4 = c1 and c6
    case5 = c3 and c6
    case6 = c3 and c2
    case7 = c5 and c4

    r1 = (m[0] - n[0])/2
    r2 = (m[0] + n[0])/2
    s1 = (m[1] - n[1])/2
    s2 = (m[1] + n[1])/2

    if case1 or case2 or case3:
        r1 = np.ceil(r1)
        r2 = np.ceil(r2)
        s1 = np.ceil(s1)
        s2 = np.ceil(s2)
    elif case4 or case5 or case6:
        r1 = np.ceil(r1)
        r2 = np.ceil(r2)
    elif case7:
        s1 = np.ceil(s1)
        s2 = np.ceil(s2)

    In = np.ones(n[1])
    P = np.arange(r1, r2)
    Q1 = list(np.kron(In, P).astype('int'))
    In = np.ones(n[0])
    P = np.arange(s1, s2)
    Q2 = list(np.kron(P, In).astype('int'))

    sampling_points = [Q1, Q2]

    return sampling_points


class Embedding(odl.Operator):

    def __init__(self, domain, range, sampling_points=None, adjoint=None):
        """Initialize a new instance.
        """
        if np.any([m > n for m, n in zip(domain.shape, range.shape)]):
            raise ValueError('The size of the domain {} is not "smaller" than '
                             'the size of the range {}'
                             .format(domain.shape, range.shape))

        self.__sampling_points = sampling_points
        self.__adjoint = adjoint

        super().__init__(domain=domain, range=range, linear=True)

        self.scale = self.domain.cell_volume / self.range.cell_volume

    @property
    def sampling_points(self):
        if self.__sampling_points is None:
            self.__sampling_points = get_central_sampling_points(
                                            self.domain.shape,
                                            self.range.shape)

        return self.__sampling_points

    def _call(self, x, out):
        out[:] = 0
        out[self.sampling_points] = x.asarray().flatten('F')
        out *= self.scale

    @property
    def adjoint(self):
        if self.__adjoint is None:
            self.__adjoint = EmbeddingAdjoint(self.range, self.domain,
                                              self.sampling_points, self)

        return self.__adjoint

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r}, {!r})'.format(self.__class__.__name__,
                                             self.domain, self.range,
                                             self.sampling_points)


class EmbeddingAdjoint(odl.Operator):

    def __init__(self, domain, range, sampling_points, adjoint):
        """Initialize a new instance.
        """

        self.__sampling_points = sampling_points
        self.__adjoint = adjoint
        super().__init__(domain=domain, range=range, linear=True)

    def _call(self, x, out):
        out[:] = np.reshape(x[self.sampling_points].asarray(),
                            self.range.shape, order='F')

    @property
    def sampling_points(self):
        return self.__sampling_points

    @property
    def adjoint(self):
        return self.__adjoint

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r}, {!r})'.format(self.__class__.__name__,
                                             self.domain, self.range,
                                             self.sampling_points)


def ind_fun_circle2d(points, center=[0.0, 0.0], radius=0.5):
    x, y = points
    return ((x-center[0])**2 + (y - center[0])**2 <= radius**2).astype(float)


def gaussian2d(x, mean=[0.0, 0.0], stddev=[0.05, 0.05]):
    """
    kernel = K.element(lambda x: misc.gaussian2d(x, stddev=[0.05, 0.05]))
    """
    return np.exp(-(((x[0] - mean[0]) / stddev[0]) ** 2 +
                    ((x[1] - mean[1]) / stddev[1]) ** 2))


# Define the total variation norm ||Dx||_1
def total_variation(domain, grad=None):
    """Total variation functional.

    Parameters
    ----------
    domain : odlspace
        domain of TV functional
    grad : gradient operator, optional
        Gradient operator of the total variation functional. This may be any
        linear operator and thereby generalizing TV. default=forward
        differences with Neumann boundary conditions

    Examples
    --------
    Check that the total variation of a constant is zero

    >>> import odl.contrib.spdhg as spdhg, odl
    >>> space = odl.uniform_discr([0, 0], [3, 3], [3, 3])
    >>> tv = spdhg.total_variation(space)
    >>> x = space.one()
    >>> tv(x) < 1e-10
    """

    if grad is None:
        grad = odl.Gradient(domain, method='forward', pad_mode='symmetric')
        # grad.norm = 2 * np.sqrt(sum(1 / grad.domain.cell_sides**2))
    # else:
    #     grad = grad

    f = odl.solvers.GroupL1Norm(grad.range, exponent=2)

    return f * grad


class TotalVariationNonNegative(odl.solvers.Functional):
    """Total variation function with nonnegativity constraint and strongly
    convex relaxation.

    In formulas, this functional may represent

        alpha * |grad x|_1 + char_fun(x) + beta/2 |x|^2_2

    with regularization parameter alpha and strong convexity beta. In addition,
    the nonnegativity constraint is achieved with the characteristic function

        char_fun(x) = 0 if x >= 0 and infty else.

    Parameters
    ----------
    domain : odlspace
        domain of TV functional
    alpha : scalar, optional
        Regularization parameter, positive
    prox_options : dict, optional
        name: string, optional
            name of the method to perform the prox operator, default=FGP
        warmstart: boolean, optional
            Do you want a warm start, i.e. start with the dual variable
            from the last call? default=True
        niter: int, optional
            number of iterations per call, default=5
        p: array, optional
            initial dual variable, default=zeros
    grad : gradient operator, optional
        Gradient operator to be used within the total variation functional.
        default=see TV
    """

    def __init__(self, domain, alpha=1, prox_options={}, grad=None,
                 strong_convexity=0, constrain='Box'):
        """
        """

        self.strong_convexity = strong_convexity
        self.constrain = constrain

        if 'name' not in prox_options:
            prox_options['name'] = 'FGP'
        if 'warmstart' not in prox_options:
            prox_options['warmstart'] = True
        if 'niter' not in prox_options:
            prox_options['niter'] = 5
        if 'p' not in prox_options:
            prox_options['p'] = None
        if 'tol' not in prox_options:
            prox_options['tol'] = None

        self.prox_options = prox_options

        self.alpha = alpha
        self.tv = total_variation(domain, grad=grad)
        self.grad = self.tv.right
        if self.constrain == 'Box':
            self.nn = odl.solvers.IndicatorBox(domain, 0, np.inf)
        elif self.constrain == 'Simplex':
            self.nn = odl.solvers.IndicatorSimplex(domain)
        else:
            raise NotImplementedError('mode {} not defined'
                                      .format(self.__constrain))
        self.l2 = 0.5 * odl.solvers.L2NormSquared(domain)
        self.proj_P = self.tv.left.convex_conj.proximal(0)
        self.proj_C = self.nn.proximal(1)

        super().__init__(space=domain, linear=False, grad_lipschitz=0)

    def __call__(self, x):
        """Evaluate functional.

        Examples
        --------
        Check that the total variation of a constant is zero

        >>> import odl.contrib.spdhg as spdhg, odl
        >>> space = odl.uniform_discr([0, 0], [3, 3], [3, 3])
        >>> tvnn = spdhg.TotalVariationNonNegative(space, alpha=2)
        >>> x = space.one()
        >>> tvnn(x) < 1e-10

        Check that negative functions are mapped to infty

        >>> import odl.contrib.spdhg as spdhg, odl, numpy as np
        >>> space = odl.uniform_discr([0, 0], [3, 3], [3, 3])
        >>> tvnn = spdhg.TotalVariationNonNegative(space, alpha=2)
        >>> x = -space.one()
        >>> np.isinf(tvnn(x))
        """

        nn = self.nn(x)

        if nn is np.inf:
            return nn
        else:
            out = self.alpha * self.tv(x) + nn
            if self.strong_convexity > 0:
                out += self.strong_convexity * self.l2(x)
            return out

    def proximal(self, sigma):
        """Prox operator of TV. It allows the proximal step length to be a
        vector of positive elements.

        Examples
        --------
        Check that the proximal operator is the identity for sigma=0

        >>> import odl.contrib.solvers.spdhg as spdhg, odl, numpy as np
        >>> space = odl.uniform_discr([0, 0], [3, 3], [3, 3])
        >>> tvnn = spdhg.TotalVariationNonNegative(space, alpha=2)
        >>> x = -space.one()
        >>> y = tvnn.proximal(0)(x)
        >>> (y-x).norm() < 1e-10

        Check that negative functions are mapped to 0

        >>> import odl.contrib.solvers.spdhg as spdhg, odl, numpy as np
        >>> space = odl.uniform_discr([0, 0], [3, 3], [3, 3])
        >>> tvnn = spdhg.TotalVariationNonNegative(space, alpha=2)
        >>> x = -space.one()
        >>> y = tvnn.proximal(0.1)(x)
        >>> y.norm() < 1e-10
        """

        if sigma == 0:
            return odl.IdentityOperator(self.domain)

        else:
            def tv_prox(z, out=None):

                if out is None:
                    out = z.space.zero()

                opts = self.prox_options

                sigma_ = np.copy(sigma)
                z_ = z.copy()

                if self.strong_convexity > 0:
                    sigma_ /= (1 + sigma * self.strong_convexity)
                    z_ /= (1 + sigma * self.strong_convexity)

                if opts['name'] == 'FGP':
                    if opts['warmstart']:
                        if opts['p'] is None:
                            opts['p'] = self.grad.range.zero()

                        p = opts['p']
                    else:
                        p = self.grad.range.zero()

                    sigma_sqrt = np.sqrt(sigma_)

                    z_ /= sigma_sqrt
                    grad = sigma_sqrt * self.grad
                    grad.norm = sigma_sqrt * self.grad.norm(estimate=True)
                    niter = opts['niter']
                    alpha = self.alpha
                    out[:] = fgp_dual(p, z_, alpha, niter, grad, self.proj_C,
                                      self.proj_P, tol=opts['tol'])

                    out *= sigma_sqrt

                    return out

                else:
                    raise NotImplementedError('Not yet implemented')

            return tv_prox


def fgp_dual(p, data, alpha, niter, grad, proj_C, proj_P, tol=None, **kwargs):
    """Computes a solution to the ROF problem with the fast gradient
    projection algorithm.

    Parameters
    ----------
    p : np.array
        dual initial variable
    data : np.array
        noisy data / proximal point
    alpha : float
        regularization parameter
    niter : int
        number of iterations
    grad : instance of gradient class
        class that supports grad(x), grad.adjoint(x), grad.norm
    proj_C : function
        projection onto the constraint set of the primal variable,
        e.g. non-negativity
    proj_P : function
        projection onto the constraint set of the dual variable,
        e.g. norm <= 1
    tol : float (optional)
        nonnegative parameter that gives the tolerance for convergence. If set
        None, then the algorithm will run for a fixed number of iterations

    Other Parameters
    ----------------
    callback : callable, optional
        Function called with the current iterate after each iteration.
    """

    # Callback object
    callback = kwargs.pop('callback', None)
    if callback is not None and not callable(callback):
        raise TypeError('`callback` {} is not callable'.format(callback))

    factr = 1 / (grad.norm**2 * alpha)

    q = p.copy()
    x = data.space.zero()

    t = 1.
    global kt

    if tol is None:
        def convergence_eval(p1, p2):
            return False
    else:
        def convergence_eval(p1, p2):
            return (p1 - p2).norm() < tol * p1.norm()

    pnew = p.copy()

    if callback is not None:
        callback(p)

    for k in range(niter):
        t0 = t
        grad.adjoint(q, out=x)
        proj_C(data - alpha * x, out=x)
        grad(x, out=pnew)
        pnew *= factr
        pnew += q

        proj_P(pnew, out=pnew)

        converged = convergence_eval(p, pnew)

        if not converged or k <= 8:
            # update step size
            t = (1 + np.sqrt(1 + 4 * t0 ** 2)) / 2.

            # calculate next iterate
            q[:] = pnew + (t0 - 1) / t * (pnew - p)

        p[:] = pnew
        kt = k + 1

        if converged and k > 8:
            t = None
            kt = k + 1
            break

        if callback is not None:
            callback(p)

    # get current image estimate
    x = proj_C(data - alpha * grad.adjoint(p))

    return x


# This is not good but can't think of anything better right now
class DataFitL2LinearPlusConv(odl.solvers.Functional):
    """Data fit functional with L2 norm for a linear operator applied to the
    convolution.

    In formulas, this functional may represent

    .. math::
        |A (x * k) - data|_2^2

    Parameters
    ----------
    domain : odlspace
        domain of the functional
    linear operator : The operator ``A`` in the formula
        Operator to be applied to the convolution
    data : in range of ``A`
    """

    def __init__(self, domain, linear_operator, data):
        if not len(domain) == 2:
            raise ValueError('Domain has not the right shape. Len=2 expected')

        self.linear_operator = linear_operator
        self.data = data
        norm = odl.solvers.L2NormSquared(data.space)
        self.data_fit = 0.5 * norm.translated(data)
        self.dom = domain

        super().__init__(domain, linear=False)

    def __call__(self, x):
        conv_operator = Convolution(self.domain[0], x[1])
        tmp_u = conv_operator(x[0])
        return self.data_fit(self.linear_operator(tmp_u))

    def partial_derivative(self, i):

        forward_model = self

        class auxOperator(Operator):

            def __init__(self, i):

                self.i = i

                super(auxOperator, self).__init__(forward_model.dom,
                                                  forward_model.dom[i])

            def _call(self, x, out=None):

                if out is None:
                    if self.i == 0:
                        out = forward_model.dom[0].zero()
                    elif self.i == 1:
                        out = forward_model.dom[1].zero()

                A = forward_model.linear_operator
                C1 = Convolution(forward_model.domain[0], x[1])

                if self.i == 0:
                    d0 = C1.adjoint(A.adjoint(A(C1(x[0])) - forward_model.data))
                    #return np.sqrt(self.domain[0].cell_volume) * d0
                    out.assign(np.sqrt(forward_model.dom[0].cell_volume) * d0)

                if self.i == 1:
                    # J = Embedding(self.domain[1], self.domain[0])
                    # C0 = Convolution(self.domain[0], x[0]) * J
                    C0 = ConvolutionEmbedding(forward_model.dom[1], forward_model.dom[0], x[0])
                    d1 = C0.adjoint(A.adjoint(A(C1(x[0])) - forward_model.data))
                    #return np.sqrt(self.domain[1].cell_volume) * d1
                    out.assign(np.sqrt(forward_model.dom[1].cell_volume) * d1)

        return auxOperator(i)

        #     A = self.linear_operator
        #     C1 = Convolution(self.domain[0], x[1])
        #
        #     if i == 0:
        #         d0 = C1.adjoint(A.adjoint(A(C1(x[0])) - self.data))
        #         return np.sqrt(self.domain[0].cell_volume) * d0
        #
        #     if i == 1:
        #         # J = Embedding(self.domain[1], self.domain[0])
        #         # C0 = Convolution(self.domain[0], x[0]) * J
        #         C0 = ConvolutionEmbedding(self.domain[1], self.domain[0], x[0])
        #         d1 = C0.adjoint(A.adjoint(A(C1(x[0])) - self.data))
        #         return np.sqrt(self.domain[1].cell_volume) * d1
        #
        # return partial_deriv

    # def gradient(self, x):
    #         A = self.linear_operator
    #         # J = Embedding(self.domain[1], self.domain[0])
    #         # C0 = Convolution(self.domain[0], x[0]) * J
    #         C0 = ConvolutionEmbedding(self.domain[1], self.domain[0], x[0])
    #         C1 = Convolution(self.domain[0], x[1])
    #         aux = A.adjoint(A(C1(x[0])) - self.data)
    #         d0 = np.sqrt(self.domain[0].cell_volume) * C1.adjoint(aux)
    #         d1 = np.sqrt(self.domain[1].cell_volume) * C0.adjoint(aux)
    #         grad_list = [d0, d1]
    #         return self.domain.element(grad_list)

    @property
    def gradient(self):
        return odl.BroadcastOperator(*[self.partial_derivative(i)
                                   for i in range(2)])


class DataFitL2LinearPlusConvViaFFT(odl.solvers.Functional):
    """Data fit functional with L2 norm for a linear operator applied to the
    convolution.

    In formulas, this functional may represent

    .. math::
        |A (x * k) - data|_2^2

    Parameters
    ----------
    domain : odlspace
        domain of the functional
    linear operator : The operator ``A`` in the formula
        Operator to be applied to the convolution
    data : in range of ``A`
    """

    def __init__(self, domain, linear_operator, data):
        if not len(domain) == 2:
            raise ValueError('Domain has not the right shape. Len=2 expected')

        self.linear_operator = linear_operator
        self.data = data
        norm = odl.solvers.L2NormSquared(data.space)
        self.data_fit = 0.5 * norm.translated(data)
        self.dom = domain

        super().__init__(domain, linear=False)

    def __call__(self, x):
        conv_operator = ConvolutionViaFFT(self.domain[0], x[1])
        tmp_u = conv_operator(x[0])
        return self.data_fit(self.linear_operator(tmp_u))

    def partial_derivative(self, i):

        forward_model = self

        class auxOperator(Operator):

            def __init__(self, i):

                self.i = i

                super(auxOperator, self).__init__(forward_model.dom,
                                                  forward_model.dom[i])

            def _call(self, x, out=None):

                if out is None:
                    if self.i == 0:
                        out = forward_model.dom[0].zero()
                    elif self.i == 1:
                        out = forward_model.dom[1].zero()

                A = forward_model.linear_operator
                C1 = ConvolutionViaFFT(forward_model.domain[0], x[1])

                if self.i == 0:
                    d0 = C1.adjoint(A.adjoint(A(C1(x[0])) - forward_model.data))
                    #return np.sqrt(self.domain[0].cell_volume) * d0
                    out.assign(np.sqrt(forward_model.dom[0].cell_volume) * d0)

                if self.i == 1:
                    J = Embedding(self.domain[1], self.domain[0])
                    C0 = ConvolutionViaFFT(self.domain[0], x[0]) * J
                    # C0 = ConvolutionEmbeddingViaFFT(forward_model.dom[1], forward_model.dom[0], x[0])
                    d1 = C0.adjoint(A.adjoint(A(C1(x[0])) - forward_model.data))
                    #return np.sqrt(self.domain[1].cell_volume) * d1
                    out.assign(np.sqrt(forward_model.dom[1].cell_volume) * d1)

        return auxOperator(i)

    @property
    def gradient(self):
        return odl.BroadcastOperator(*[self.partial_derivative(i)
                                   for i in range(2)])


class fbs():

    r"""Forward Backward Splitting algorithm

    First order primal-dual hybrid-gradient method for non-smooth convex
    optimization problems with known saddle-point structure. The
    primal formulation of the general problem is

    .. math::
        \min_{x = (u,k) in X = U \times K} Rk(k) + Ru(u) + g(A (x * k))

    where ``A`` is an operator and ``Rk`` and ``Ru`` are functionals.
    Here :math:`g(y) = \|y - data\|^2_2

    Parameters
    ----------
    domain : ProductSpace
        Minimization space
    A : linear `Operator`
        The operator ``A`` in the problem definition. Needs to have
        ``A.adjoint``.
    g : `Functional`
        The function ``g`` in the problem definition.
    Ru : `Functional` in the problem definition.
        Regularizer of variable u
    Rk : `Functional` in the problem definition.
        Regularizer of variable k
    x : ``X.domain`` element
        Starting point of the iteration, updated in-place.
    niter : non-negative int
        Number of iterations.

    """

    """ Code without class

    for i in range(niter):

                xt = Reg.proximal(sigma)(x - sigma * g.gradient(x))

                norm_2 = 0.5 * L2NormSquared(Umr).translated(x)
                while g(xt) > (g(x) + g.gradient(x).inner(xt-x) + L * 0.5 *
                               ((xt - x).norm())**2):
                    L = 2 * L
                    sigma = 1/L
                    xt = Reg.proximal(sigma)(x - sigma * g.gradient(x))
                x = xt
                L = 0.9 * L
                sigma = 1/L
                cb(x)
    """

    def __init__(self, domain, g, Reg, sigma, g_grad=None, x=None, niter=None,
                 callback=None, txt_file_L=None):

        self.domain = domain
        self.g = g
        self.Reg = Reg
        self.sigma = sigma
        self.L = 1/sigma
        self.xt = self.domain.zero()
        self.callback = callback
        self.txt_file_L = txt_file_L
        self.g_grad = g_grad

        if x is None:
            x = self.domain.one()

        self.x = x
        self.x_ = x.copy()

        if g_grad is None:
            g_grad = g.gradient
        else:
            g_grad = g_grad

        self.g_grad = g_grad

        # if line_search is None:
        #    line_search = BacktrackingLineSearch(f + g)

        if niter is not None:
            self.run(niter)

    def function_value(self, x):
        return self.g(x) + self.Reg(x)

    def backtracking(self):

        g = self.g
        x = self.x
        xt = self.xt
        self.x_ = x.copy()
        Reg = self.Reg
        sigma = self.sigma
        L = self.L
        g_grad = self.g_grad
        g_x = g(x)
        g_grad_x = g_grad(x)

        while g(xt) > (g_x + g_grad_x.inner(xt-x) + L * 0.5 *
                       ((xt - x).norm())**2):
            L *= 2
            sigma = 1/L
            Reg.proximal(sigma)(x - sigma * g_grad_x, out=xt)

        x.assign(xt)
        L *= 0.9
        sigma = 1/L
        self.x = x
        self.sigma = sigma
        self.xt = xt
        self.L = L

    def update(self):

        x = self.x
        Reg = self.Reg
        sigma = self.sigma
        xt = self.xt
        g_grad = self.g_grad
        txt_file_L = self.txt_file_L

        g_grad_x = g_grad(x)

        Reg.proximal(sigma)(x - sigma * g_grad_x, out=xt)

        self.backtracking()
        if txt_file_L is not None:
            if os.path.isfile(txt_file_L):
                file = open(txt_file_L, 'a')
                file.write(str(1/sigma) + ' ')
                file.close()

    def run(self, niter=1):
        for i in range(niter):
            if self.callback is not None:
                self.callback(self.x)
            self.update()
            f1 = self.function_value(self.x)
            f2 = self.function_value(self.x_)
            if (np.abs(f1 - f2) < 1e-6 * f1):
                self.callback(self.x)
                print('The algorithm converges')
                break


class bregman_iteration():

    r"""Specialised linearised Bregmann iteration for minimising
    E(u) = \|R(u * k) - data\|^2

    First order primal-dual hybrid-gradient method for non-smooth convex
    optimization problems with known saddle-point structure. The
    primal formulation of the general problem is

    .. math::
        \min_{x = (u,k) in X = U \times K} Rk(k) + Ru(u) + g(A (x * k))

    where ``A`` is an operator and ``Rk`` and ``Ru`` are functionals.
    Here :math:`g(y) = \|y - data\|^2_2

    Parameters
    ----------
    domain : ProductSpace
        Minimization space
    A : linear `Operator`
        The operator ``A`` in the problem definition. Needs to have
        ``A.adjoint``.
    g : `Functional`
        The function ``g`` in the problem definition.
    Ru : `Functional` in the problem definition.
        Regularizer of variable u
    Rk : `Functional` in the problem definition.
        Regularizer of variable k
    x : ``X.domain`` element
        Starting point of the iteration, updated in-place.
    niter : non-negative int
        Number of iterations.

    """

    def __init__(self, domain, g, Reg, sigma, lc, g_grad=None, x=None, q=None,
                 niter=None, callback=None, txt_file_L=None, alg=None):

        self.domain = domain
        self.g = g
        self.Reg = Reg
        self.sigma = sigma
        self.L = 1/sigma
        self.xt = self.domain.zero()
        self.callback = callback
        self.txt_file_L = txt_file_L
        self.alg = alg
        self.lc = lc

        if g_grad is None:
            g_grad = g.gradient

        self.g_grad = g_grad

        if x is None:
            x = self.domain.zero()

        if q is None:
            q = self.domain.zero()

        self.x = x
        self.q = q
        self.x_ = x.copy()

        # if line_search is None:
        #    line_search = BacktrackingLineSearch(f + g)

        if niter is not None:
            self.run(niter)

    def function_value(self, x):
        return self.g(x)

    def backtracking(self):

        g = self.g
        x = self.x
        q = self.q
        xt = self.xt
        Reg = self.Reg
        sigma = self.sigma
        L = self.L
        g_grad = self.g_grad
        x_ = self.x_
        g_x = g(x)
        g_grad_x = g_grad(x)

        while g(xt) > (g_x + g_grad_x.inner(xt-x) + L * 0.5 *
                       ((xt - x).norm())**2):
            L *= 2
            sigma = 1/L
            Reg.proximal(sigma)(x + sigma * (q - g_grad_x), out=xt)

        x.assign(xt)
        q += - L * (x - x_ + sigma * g.gradient(x_))
        L *= 0.9
        sigma = 1/L
        self.L = L
        self.sigma = sigma
        self.x_ = x.copy()

    def update(self):

        x = self.x
        x_ = self.x_
        q = self.q
        Reg = self.Reg
        sigma = self.sigma
        L = self.L
        xt = self.xt
        g = self.g
        g_grad = self.g_grad
        g_x = g(x)
        g_grad_x = g_grad(x)
        txt_file_L = self.txt_file_L
        # alg = self.alg

        Reg.proximal(sigma)(x + sigma * (q - g_grad_x), out=xt)

        # self.backtracking()
        while g(xt) > (g_x + g_grad_x.inner(xt - x) + L * 0.5 *
                       ((xt - x).norm())**2):
            L *= 2
            sigma = 1/L
            Reg.proximal(sigma)(x + sigma * (q - g_grad_x), out=xt)

        # fname = '{}/inner_iter_{}.txt'.format(sfolder_txt, alg)
        # if os.path.isfile(fname):
        #     file = open(fname, 'a')
        #     file.write(str(kt) + ' ')
        #     file.close()

        x.assign(xt)
        q += - L * (x - x_ + sigma * g.gradient(x_))
        L *= 0.9
        sigma = 1/L
        self.L = L
        self.sigma = sigma
        self.x_ = x.copy()

        if os.path.isfile(txt_file_L):
            file = open(txt_file_L, 'a')
            file.write(str(L) + ' ')
            file.close()

#    def run(self, niter=1):
#        for i in range(niter):
#            if self.callback is not None:
#                self.callback(self.x)
#
#            self.update()

    def run(self, niter=1):
        for i in range(niter):
            if self.callback is not None:
                self.callback(self.x)

            # self.update()
            f1 = self.function_value(self.x)
            if (f1 < self.lc):
                self.callback(self.x)
                print('The algorithm converges')
                break
            else:
                self.update()


# class PALM():
#
#     r"""Proximal Alternating Linearized Minimization algorithm.
#
#     First order primal-dual hybrid-gradient method for non-smooth convex
#     optimization problems with known saddle-point structure. The
#     primal formulation of the general problem is
#
#     .. math::
#         \min_{x = (u,k) in X = U \times K} Rk(k) + Ru(u) + g(A (x * k))
#
#     where ``A`` is an operator and ``Rk`` and ``Ru`` are functionals.
#     Here :math:`g(y) = \|y - data\|^2_2
#
#     Parameters
#     ----------
#     domain : ProductSpace
#         Minimization space
#     A : linear `Operator`
#         The operator ``A`` in the problem definition. Needs to have
#         ``A.adjoint``.
#     g : `Functional`
#         The function ``g`` in the problem definition.
#     Ru : `Functional` in the problem definition.
#         Regularizer of variable u
#     Rk : `Functional` in the problem definition.
#         Regularizer of variable k
#     x : ``X.domain`` element
#         Starting point of the iteration, updated in-place.
#     niter : non-negative int
#         Number of iterations.
#
#     Example:
#         palm = misc.PALM(X, g, Reg, sigma, x=x, niter=niter, callback=cb)
#     """
#
#     def __init__(self, domain, g, Reg, sigma, g_grad=None, x=None, niter=None,
#                  callback=None, txt_file_L=None, alg=None):
#
#         self.domain = domain
#         self.g = g
#         self.Reg = Reg
#         self.sigma = sigma
#         self.xt = self.domain.zero()
#         self.callback = callback
#         self.g_grad = g_grad
#         self.txt_file_L = txt_file_L
#         self.alg = alg
#
#         if x is None:
#             x = self.domain.one()
#             x[1] = self.Reg[1].proximal(1)(x[1])
#
#         self.x = x
#         self.x_ = x.copy()
#
#         # if line_search is None:
#         #    line_search = BacktrackingLineSearch(f + g)
#
#         if niter is not None:
#             self.run(niter)
#
#     def backtracking(self, i):
#
#         X = self.domain
#         g = self.g
#         x = self.x
#         xt = self.xt
#         if self.g_grad is None:
#             g_grad_x = g.partial_derivative(x, i)
#         else:
#             g_grad_x = self.g_grad[i](x)
#         Reg = self.Reg
#         sigma = self.sigma
#
#         norm_2 = 0.5 * odl.solvers.L2NormSquared(X[i]).translated(x[i])
#         while g(xt) > (g(x) + g_grad_x.inner(xt[i]-x[i]) +
#                        (1/sigma[i]) * 0.5 * norm_2(xt[i])):
#             sigma[i] *= 0.5
#             # print(sigma)
#             Reg[i].proximal(sigma[i])(x[i] - sigma[i] * g_grad_x, out=xt[i])
#
#         self.sigma = sigma
#
#     def function_value(self, x):
#         return self.g(x) + self.Reg(x)
#
#     def update(self):
#
#         X = self.domain
#         x = self.x
#         g = self.g
#         Reg = self.Reg
#         sigma = self.sigma
#         xt = self.xt
#         txt_file_L = self.txt_file_L
#         # alg = self.alg
#         self.x_ = x.copy()
#
#         for i in range(len(X)):
#
#             if self.g_grad is None:
#                 g_grad_x = g.partial_derivative(x, i)
#             else:
#                 g_grad_x = self.g_grad[i](x)
#
#             Reg[i].proximal(sigma[i])(x[i] - sigma[i] * g_grad_x, out=xt[i])
#
#             # self.backtracking(i)
#             g_x = g(x)
#
#             while g(xt) > (g_x + g_grad_x.inner(xt[i]-x[i]) +
#                            (1/sigma[i]) * 0.5 * ((xt[i] - x[i]).norm())**2):
#                 sigma[i] *= 0.5
#                 print('backtracking sigma_{} = {}'.format(i, sigma[i]))
#                 Reg[i].proximal(sigma[i])(x[i] - sigma[i] * g_grad_x,
#                                           out=xt[i])
#
#             x[i].assign(self.xt[i])
#             sigma[i] *= 10/9
#             self.sigma = sigma
#             print('outside sigma_{} = {}'.format(i, sigma[i]))
#
#         txt_file_Lu = txt_file_L.replace('Lipschitz', 'Lipschitz_u')
#         txt_file_Lk = txt_file_L.replace('Lipschitz', 'Lipschitz_k')
#
#         if os.path.isfile(txt_file_Lu):
#             file = open(txt_file_Lu, 'a')
#             file.write(str(1/sigma[0]) + ' ')
#             file.close()
#
#         if os.path.isfile(txt_file_Lk):
#             file = open(txt_file_Lk, 'a')
#             file.write(str(1/sigma[1]) + ' ')
#             file.close()
#
#         # fname = '{}/inner_iter_{}.txt'.format(sfolder_txt, alg)
#         # if os.path.isfile(fname):
#         #     file = open(fname, 'a')
#         #     file.write(str(kt) + ' ')
#         #     file.close()
#
#     def run(self, niter=1):
#         for i in range(niter):
#             if self.callback is not None:
#                 self.callback(self.x)
#             self.update()
#             f1 = self.function_value(self.x)
#             f2 = self.function_value(self.x_)
#             if (np.abs(f1 - f2) < 1e-6 * f1):
#                 self.callback(self.x)
#                 print('The algorithm converges')
#                 break

class PALM():

    def __init__(self, f, g, ud_vars=None, x=None, niter=None,
                 callback=None, L=None, tol=None):

        if x is None:
            x = f.domain.zero()

        if L is None:
            L = [1e2] * len(x)

        if ud_vars is None:
            ud_vars = range(len(x))

        self.ud_vars = ud_vars
        self.x = x
        self.f = f
        self.g = g
        self.etas = [0.5, 10]  # [0.5, 10]
        self.L = L
        self.tol = tol
        self.callback = callback
        self.niter = 0

        self.dx = None
        self.x_old = None
        self.x_old2 = None

        self.g_old = None
        self.f_old = None

        if niter is not None:
            self.run(niter)

    def update_coordinate(self, i):

        x = self.x
        x_old = self.x_old
        f = self.f
        g = self.g
        L = self.L

        BTsuccess = False
        if i == 0:
            bt_vec = range(60)
        else:
            bt_vec = range(60)

        l2sq = odl_l2sq(x[i].space)
        df = f.gradient[i](x_old)

        t0 = time()
        for bt in bt_vec:  # backtracking loop

            g[i].proximal(1 / L[i])(x_old[i] - 1 / L[i] * df, out=x[i])

            # backtracking on Lipschitz constants
            f_new = f(x)
            LHS1 = f_new

            self.dx[i] = x[i] - x_old[i]

            df_dxi = df.inner(self.dx[i])
            dxi_sq = l2sq(self.dx[i])

            RHS1 = self.f_old + df_dxi + L[i] / 2 * dxi_sq

            eps = 0e-4

            # print(i, bt, LHS1 - RHS1)
            if LHS1 > RHS1 + eps:
                L[i] *= self.etas[1]
                continue

            # proximal backtracking
            gi_new = g[i](x[i])
            LHS2 = gi_new
            RHS2 = self.g_old[i] - df_dxi - L[i] / 2 * dxi_sq

            if LHS2 <= RHS2 + eps:
                x_old[i][:] = x[i]
                self.f_old = f_new
                self.g_old[i] = gi_new
                L[i] *= self.etas[0]
                BTsuccess = True
                break

            L[i] *= self.etas[1]

        if BTsuccess is False:
            print('No step size found for variable {} after {} backtracking steps'.format(i, bt))

        if self.tol is not None:
            reldiff = dxi_sq / max(l2sq(x[i]), 1e-4)

            if reldiff < self.tol:
                self.ud_vars.remove(i)
                print('Variable {} stopped updating'.format(i))

        dt = time() - t0

        print("time updating component "+str(i)+":"+str(dt))

    def update(self):
        self.niter += 1

        if self.dx is None:
            self.dx = self.x.copy()
        if self.x_old is None:
            self.x_old = self.x.copy()
        if self.f_old is None:
            self.f_old = self.f(self.x_old)
        if self.g_old is None:
            self.g_old = [self.g[j](self.x_old[j]) for j in range(len(self.x))]

        for i in self.ud_vars:  # loop over variables

            self.update_coordinate(i)

    def run(self, niter=1):
        if self.tol is not None:
            if self.x_old2 is None:
                self.x_old2 = self.x.copy()
            l2sq = odl_l2sq(self.x.space)

        for k in range(niter):
            if self.x_old2 is None:
                self.x_old2 = self.x.copy()
            self.x_old2[:] = self.x
            self.update()

            dx = []
            for i in range(len(self.x)):
                l2sq = odl_l2sq(self.x[i].space)
                dx.append(l2sq(self.dx[i]) / max(l2sq(self.x[i]), 1e-4))

            s = 'obj:{:3.2e}, f:{:3.2e}, g:{:3.2e}, diff:' + '{:3.2e} ' * len(self.x) + 'lip:' + '{:3.2e} ' * len(
                self.x)

            fx = self.f(self.x)
            gx = self.g(self.x)
            print(s.format(fx + gx, fx, gx, *dx, *self.L))

            if self.callback is not None:
                self.callback(self.x)

            if self.tol is not None:
                l2sq = odl_l2sq(self.x.space)
                norm = l2sq(self.x_old2)
                if k > 1 and norm > 0:
                    crit = l2sq(self.x_old2 - self.x) / norm
                else:
                    crit = np.inf

                if crit < self.tol:
                    print('Stopped iterations with rel. diff. ', crit)
                    break
                else:
                    self.x_old2[:] = self.x

        return self.x

class blind_bregman_iteration():

    def __init__(self, domain, g, Reg, sigma, lc, g_grad=None, x=None,
                 q=None, niter=None, callback=None, txt_file_L=None,
                 alg=None):

        self.domain = domain
        self.g = g
        self.Reg = Reg
        self.sigma = sigma
        self.L = 1/self.sigma
        self.xt = self.domain.zero()
        self.callback = callback
        self.g_grad = g_grad
        self.txt_file_L = txt_file_L
        self.alg = alg
        self.lc = lc

        if x is None:
            x = self.domain.zero()
            x[1] = self.Reg[1].proximal(1)(x[1])

        if q is None:
            q = self.domain.zero()
        self.x = x
        self.q = q
        self.x_ = x.copy()

        # if line_search is None:
        #    line_search = BacktrackingLineSearch(f + g)

        if niter is not None:
            self.run(niter)

    def function_value(self, x):
        return self.g(x)

    def update(self):

        X = self.domain
        x = self.x
        x_ = self.x_
        q = self.q
        g = self.g
        Reg = self.Reg
        sigma = self.sigma
        L = self.L
        xt = self.xt
        txt_file_L = self.txt_file_L
        # alg = self.alg

        for i in range(len(X)):

            if self.g_grad is None:
                g_grad_x = g.partial_derivative(x, i)
            else:
                g_grad_x = self.g_grad[i](x)

            Reg[i].proximal(sigma[i])(x[i] + sigma[i] * (q[i] - g_grad_x),
                                      out=xt[i])

            # self.backtracking(i)
            g_x = g(x)

            while g(xt) > (g_x + g_grad_x.inner(xt[i]-x[i]) +
                           L[i] * 0.5 * ((xt[i] - x[i]).norm())**2):
                L[i] *= 2
                # print(sigma)
                sigma = 1/L
                Reg[i].proximal(sigma[i])(x[i] + sigma[i] * (q[i] - g_grad_x),
                                          out=xt[i])

            x[i].assign(self.xt[i])
            q[i] += - L[i] * (x[i] - x_[i] + sigma[i] * g_grad_x)
            q[i].assign(self.q[i])
            L[i] *= 0.9
            sigma = 1/L
            self.L = L
            self.sigma = sigma
            self.x_ = x.copy()

            txt_file_Lu = txt_file_L.replace('Lipschitz', 'Lipschitz_u')
            txt_file_Lk = txt_file_L.replace('Lipschitz', 'Lipschitz_k')

            if os.path.isfile(txt_file_Lu):
                file = open(txt_file_Lu, 'a')
                file.write(str(1/sigma[0]) + ' ')
                file.close()

            if os.path.isfile(txt_file_Lk):
                file = open(txt_file_Lk, 'a')
                file.write(str(1/sigma[1]) + ' ')
                file.close()

        # fname = '{}/inner_iter_{}.txt'.format(sfolder_txt, alg)
        # if os.path.isfile(fname):
        #     file = open(fname, 'a')
        #     file.write(str(kt) + ' ')
        #     file.close()

    def run(self, niter=1):
        for i in range(niter):
            if self.callback is not None:
                self.callback(self.x)
            f1 = self.function_value(self.x)
            if (f1 < self.lc):
                self.callback(self.x)
                print('The algorithm converges')
                break
            else:
                self.update()


def test_adjoint(A):

    kd = odl.phantom.uniform_noise(A.domain)
    kr = odl.phantom.uniform_noise(A.range)

    ip3 = A(kd).inner(kr)
    ip4 = A.adjoint(kr).inner(kd)

    print(ip3/ip4)


def save_png(image, name, folder):
    clim = [0, 1]
    m = (clim[1] - clim[0])/(np.max(image) - np.min(image))
    x = m * (image - np.min(image)) + clim[0]
    m = 1/(np.max(x) - np.min(x))
    x = m * (x - np.min(x))
    imsave('{}/{}.png'.format(folder, name), np.rot90(x, 1))


def save_sinogram(sinogram, name, folder):
    plt.close()
    plt.imshow(sinogram, cmap=plt.cm.gray, aspect=6)
    plt.axis('off')
    plt.colorbar()
    plt.savefig('{}/{}.pdf'.format(folder, name),
                format='pdf', dpi=1000, bbox_inches='tight', pad_inches=0.05)


def save_image(image, name, folder, fignum, ssim=None, psnr=None,
               hpsi=None, obj=None, niter=None, cmap='gray', vmin=0,
               vmax=None):

    # matplotlib.rc('text', usetex=False)
    # plt.rc('text', usetex=True)

    fig = plt.figure(fignum)
    plt.clf()
    img = plt.imshow(np.rot90(image, -1), cmap=cmap, vmin=vmin,
                     vmax=vmax)
    matplotlib.pyplot.colorbar(aspect=25, pad=0.03)
    img.axes.get_xaxis().set_ticks([])
    img.axes.get_yaxis().set_ticks([])

    if ssim is not None:

        plt.xlabel("iter={0},      f(x)={1:.4g}"
                   "\n"
                   " SSIM={2:.3g}, PSNR={3:.3g}, HPSI={4:.3g}"
                   .format(niter, obj, ssim, psnr, hpsi))
    # plt.clim(0, 261)
    fig.savefig('{}/{}.pdf'.format(folder, name),
                format='pdf', dpi=1000, bbox_inches='tight', pad_inches=0.05)

#    if clim is not None:
#
#        m = (clim[1] - clim[0])/(np.max(image) - np.min(image))
#        x = m * (image - np.min(image)) + clim[0]
#
#        m = 1/(np.max(x) - np.min(x))
#        x = m * (x - np.min(x))
#
#        imsave('{}/{}.png'.format(folder, name), np.rot90(x, 1))


def save_kernel(image, name, folder, fignum, ssim=None, psnr=None,
                hpsi=None, obj=None, niter=None,  cmap='viridis', clim=None):

    # matplotlib.rc('text', usetex=False)
    # plt.rc('text', usetex=True)

    fig = plt.figure(fignum)
    plt.clf()
    img = plt.imshow(np.rot90(image, 1), cmap=plt.cm.viridis)
    matplotlib.pyplot.colorbar(aspect=25, pad=0.01)
    img.axes.get_xaxis().set_ticks([])
    img.axes.get_yaxis().set_ticks([])

    if ssim is not None:
        plt.xlabel("iter={0},      f(x)={1:.4g}"
                   "\n"
                   " SSIM={2:.3g}, PSNR={3:.3g}, HPSI={4:.3g}"
                   .format(niter, obj, ssim, psnr, hpsi))
    fig.savefig('{}/{}_fig.png'.format(folder, name), bbox_inches='tight')

#    if clim is None:
#        m = 1/(np.max(image) - np.min(image))
#        x = m * (image - np.min(image))
#    else:
#        m = (clim[1] - clim[0])/(np.max(image) - np.min(image))
#        x = m * (image - np.min(image)) + clim[0]
#
#    imsave('{}/{}.png'.format(folder, name), np.rot90(x, 1))


def save_ssim(image, name, folder, niter=None, ssim=None):

    fig = plt.figure(1)
    plt.clf()
    # boring_cmap = plt.cm.get_cmap("twilight")
    boring_cmap = plt.cm.get_cmap("gray")
    img = plt.imshow(np.rot90(image, 0), cmap=boring_cmap, vmin=-1, vmax=1)
    matplotlib.pyplot.colorbar(aspect=25, pad=0.01)
    img.axes.get_xaxis().set_ticks([])
    img.axes.get_yaxis().set_ticks([])

    fig.savefig('{}/{}_fig.png'.format(folder, name), bbox_inches='tight')


def dTV(U, sinfo, eta):

    grad = odl.Gradient(U)
    grad_space = grad.range

    sinfo_grad = grad(sinfo)

    norm = odl.PointwiseNorm(grad_space, 2)
    norm_sinfo_grad = norm(sinfo_grad)

    max_norm = np.max(norm_sinfo_grad)
    eta_scaled = eta * max(max_norm, 1e-4)
    norm_eta_sinfo_grad = np.sqrt(norm_sinfo_grad ** 2 +
                                  eta_scaled ** 2)  # SHOULD BE DONE BETTER
    xi = grad_space.element([g / norm_eta_sinfo_grad for g in sinfo_grad])

    Id = odl.operator.IdentityOperator(grad_space)
    xiT = odl.PointwiseInner(grad_space, xi)
    xixiT = odl.BroadcastOperator(*[x*xiT for x in xi])

    gamma = 1
    D = (Id - gamma * xixiT) * grad
    return D


class PALM_test():

    def __init__(self, domain, g, Reg, sigma, g_grad=None, x=None, niter=None,
                 callback=None, sfolder_txt=None, alg=None):

        self.domain = domain
        self.g = g
        self.Reg = Reg
        self.sigma = sigma
        self.xt = self.domain.zero()
        self.callback = callback
        self.g_grad = g_grad
        self.sfolder_txt = sfolder_txt
        self.alg = alg

        if x is None:
            x = self.domain.one()
            x[1] = self.Reg[1].proximal(1)(x[1])

        self.x = x
        self.x_ = x.copy()

        if niter is not None:
            self.run(niter)

    def backtracking(self, i):

        g = self.g
        x = self.x
        xt = self.xt
        if self.g_grad is None:
            g_grad_x = g.partial_derivative(x, i)

        Reg = self.Reg
        sigma = self.sigma

        while g(xt) > (g(x) + g_grad_x.inner(xt[i]-x[i]) +
                       (1/sigma[i]) * 0.5 * ((xt[i] - x[i]).norm())**2):
            sigma[i] *= 0.5
            # print(sigma)
            Reg[i].proximal(sigma[i])(x[i] - sigma[i] * g_grad_x, out=xt[i])

        self.sigma = sigma

    def function_value(self, x):
        return self.g(x) + self.Reg(x)

    def update(self):

        X = self.domain
        x = self.x
        g = self.g
        Reg = self.Reg
        sigma = self.sigma
        xt = self.xt
        sfolder_txt = self.sfolder_txt
        alg = self.alg
        self.x_ = x.copy()

        for i in range(len(X)):

            if self.g_grad is None:
                g_grad_x = g.partial_derivative(x, i)

            Reg[i].proximal(sigma[i])(x[i] - sigma[i] * g_grad_x, out=xt[i])

            # self.backtracking(i)
            g_x = g(x)

            while g(xt) > (g_x + g_grad_x.inner(xt[i]-x[i]) +
                           (1/sigma[i]) * 0.5 * ((xt[i] - x[i]).norm())**2):
                sigma[i] *= 0.5
                # print('backtracking sigma_{} = {}'.format(i, sigma[i]))
                Reg[i].proximal(sigma[i])(x[i] - sigma[i] * g_grad_x,
                                          out=xt[i])

            x[i].assign(self.xt[i])
            sigma[i] *= 10/9
            self.sigma = sigma
            # print('outside sigma_{} = {}'.format(i, sigma[i]))

        fname = '{}/inner_iter_{}.txt'.format(sfolder_txt, alg)
        if os.path.isfile(fname):
            file = open(fname, 'a')
            # file.write(str(kt) + ' ')
            file.close()

    def run(self, niter=1):
        for _ in range(niter):
            if self.callback is not None:
                self.callback(self.x)
            self.update()
            # f1 = self.function_value(self.x)
            # f2 = self.function_value(self.x_)
            # if (np.abs(f1 - f2) < 1e-6 * f1):
            #     self.callback(self.x)
            #     print('The algorithm converges')
            #     break
