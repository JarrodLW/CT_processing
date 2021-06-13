import odl
from odl.solvers.functional.functional import Functional
from odl.operator import Operator
import numpy as np

class CTKullbackLeibler(Functional):

    r"""A functional derived from Kullback-Leibler K(u,v) for u\propto exp(-x).

    Notes
    -----
    The functional :math:`F` with prior :math:`f>=0` is given by:

    .. math::
        F(x)
        =\int I_0\exp(-x) +  f_i\left(x_i +\log(f_i/I_0)-1\right)

    Note that we use the common definition 0 log(0) := 0.
    KL based objectives are common in MLEM optimization problems and are often
    used as data-matching term when data noise governed by a multivariate
    Poisson probability distribution is significant.
    """

    def __init__(self, space, prior=None, max_intens=None):
        """Initialize a new instance.

        Parameters
        ----------
        space : `DiscretizedSpace` or `TensorSpace`
            Domain of the functional.
        prior : ``space`` `element-like`, optional
            Depending on the context, the prior, target or data
            distribution. It is assumed to be nonnegative.
            Default: if None it is take as the one-element.

        Examples
        --------

        Define F(x) = max_intens*exp(-x)

        Test that CTKullbackLeibler(x, F(x)) = 0

        >>> space = odl.rn(3)
        >>> max_intens = 10
        >>> prior = max_intens*np.exp(-space.one())
        >>> func = CTKullbackLeibler(space, prior=prior, max_intens=max_intens)
        >>> func(space.one())
        0.0


        Test that zeros in the prior are handled correctly

        >>> prior = space.zero()
        >>> max_intens = 1
        >>> func = CTKullbackLeibler(space, prior=prior, max_intens=max_intens)
        >>> x = space.zero()
        >>> func(x)
        3.0
        """
        super(CTKullbackLeibler, self).__init__(
            space=space, linear=False, grad_lipschitz=np.nan)

        if prior is not None and prior not in self.domain:
            raise ValueError('`prior` not in `domain`'
                             ''.format(prior, self.domain))

        self.__prior = prior
        self.max_intens = max_intens
        self.space = space


    @property
    def prior(self):
        """The prior in the Kullback-Leibler functional."""
        return self.__prior

    def _call(self, x):
        """Return the CT KL-divergence in the point ``x``.

        """
        # Lazy import to improve `import odl` time
        import scipy.special

        with np.errstate(invalid='ignore', divide='ignore'):
            if self.prior is None:
                res = self.max_intens*(np.exp(-x) + (x - 1)).inner(self.domain.one())
            else:
                xlogy = scipy.special.xlogy(self.prior, self.prior / self.max_intens)
                res = (self.max_intens*np.exp(-x) + self.prior*(x - 1) + xlogy).inner(self.domain.one())

        #if not np.isfinite(res):
            # In this case, some element was less than or equal to zero
        #    return np.inf
        #else:
        #    return res
        return res


    @property
    def gradient(self):
        r"""Gradient of the CT-KL functional.

        The gradient of `CTKullbackLeibler` with ``prior`` :math:`g` is given
        as

        .. math::
            \nabla G(x) = g - I_0*exp(-x).

        """
        functional = self

        class CTKLGradient(Operator):

            """The gradient operator of this functional."""

            def __init__(self):
                """Initialize a new instance."""
                super(CTKLGradient, self).__init__(
                    functional.domain, functional.domain, linear=False)

            def _call(self, x):
                """Apply the gradient operator to the given point.

                """
                if functional.prior is None:
                    return functional.max_intens*(1 - np.exp(-x))
                else:
                    return functional.prior - functional.max_intens*np.exp(-x)

        return CTKLGradient()

    @property
    def proximal(self):

        g = self.prior
        functional = self

        if g is not None and g not in self.space:
            raise TypeError('{} is not an element of {}'.format(g, functional.space))

        class ProximalCTKL(Operator):

            """Proximal operator of the convex conjugate of the KL divergence."""

            def __init__(self, sigma):
                """Initialize a new instance.

                Parameters
                ----------
                sigma : positive float
                """
                super(ProximalCTKL, self).__init__(
                    domain=functional.space, range=functional.space, linear=False)
                self.sigma = float(sigma)

            def _call(self, x, out):
                """Return ``self(x, out=out)``."""

                #if x is out:
                    # Handle aliased `x` and `out` (need original `x` later on)
                 #   x = x.copy()
                #else:
                #    out.assign(x)

                import scipy.special

                if g is None:
                    # If g is None, it is taken as I_0 times the one element
                    lambw = scipy.special.lambertw(
                        (self.sigma*functional.max_intens) * np.exp(self.sigma*functional.max_intens - x))
                else:
                    lambw = scipy.special.lambertw(
                        self.sigma*functional.max_intens*np.exp(self.sigma*g - x))

                if not np.issubsctype(self.domain.dtype, np.complexfloating):
                    lambw = lambw.real

                prox = -np.log(lambw / (self.sigma*functional.max_intens))
                prox = x.space.element(prox)
                out.assign(prox)

        return ProximalCTKL

    @property
    def convex_conj(self):
        """The convex conjugate functional of the KL-functional."""
        return CTKullbackLeiblerConvexConj(self.domain, self.prior, self.max_intens)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r})'.format(self.__class__.__name__,
                                       self.domain, self.prior)


    #
    # def __repr__(self):
    #     """Return ``repr(self)``."""
    #     return '{}({!r}, {!r})'.format(self.__class__.__name__,
    #                                    self.domain, self.prior)

class CTKullbackLeiblerConvexConj(Functional):

    r"""A functional derived from Kullback-Leibler K(u,v) for u\propto exp(-x).

    Notes
    -----


    Note that we use the common definition 0 log(0) := 0.
    KL based objectives are common in MLEM optimization problems and are often
    used as data-matching term when data noise governed by a multivariate
    Poisson probability distribution is significant.
    """

    def __init__(self, space, prior=None, max_intens=None):
        """Initialize a new instance.

        Parameters
        ----------
        space : `DiscretizedSpace` or `TensorSpace`
            Domain of the functional.
        prior : ``space`` `element-like`, optional
            Depending on the context, the prior, target or data
            distribution. It is assumed to be nonnegative.
            Default: if None it is take as the one-element.

        Examples
        --------

        Define F(x) = max_intens*exp(-x)

        Test that CTKullbackLeibler(x, F(x)) = 0

        >>> space = odl.rn(3)
        >>> max_intens = 10
        >>> prior = max_intens*np.exp(-space.one())
        >>> func = CTKullbackLeibler(space, prior=prior, max_intens=max_intens)
        >>> func(space.one())
        0.0


        Test that zeros in the prior are handled correctly

        >>> prior = space.zero()
        >>> max_intens = 1
        >>> func = CTKullbackLeibler(space, prior=prior, max_intens=max_intens)
        >>> x = space.zero()
        >>> func(x)
        3.0
        """
        super(CTKullbackLeiblerConvexConj, self).__init__(
            space=space, linear=False, grad_lipschitz=np.nan)

        if prior is not None and prior not in self.domain:
            raise ValueError('`prior` not in `domain`'
                             ''.format(prior, self.domain))

        self.__prior = prior
        self.max_intens = max_intens
        self.space = space


    @property
    def prior(self):
        """The prior in the Kullback-Leibler functional."""
        return self.__prior

    def _call(self, x):
        """Return the CT KL-divergence in the point ``x``.

        """
        # Lazy import to improve `import odl` time
        import scipy.special

        with np.errstate(invalid='ignore', divide='ignore'):
            if self.prior is None:
                # if prior is none, set to be max_intensity,l I_0
                entr_term = self.max_intens*self.domain.one().inner(self.domain.one())
                xlogy = scipy.special.xlogy(self.max_intens - x, (self.max_intens - x)/self.max_intens)
                res = entr_term - (self.max_intens - x - xlogy).inner(self.domain.one())
            else:
                xlogy_1 = scipy.special.xlogy(self.prior, self.prior / self.max_intens)
                entr_term = (xlogy_1 - self.prior).inner(self.domain.one())
                xlogy_2 = scipy.special.xlogy(self.prior - x, (self.prior - x)/self.max_intens)
                res = entr_term - (self.prior - x - xlogy_2).inner(self.domain.one())

        if not np.isfinite(res):
            # In this case, some element was less than or equal to zero
            return np.inf
        else:
            return res

    @property
    def gradient(self):
        r"""Gradient of the CT-KL functional.

        The gradient of `CTKullbackLeibler` with ``prior`` :math:`g` is given
        as

        .. math::
            \nabla G(x) = g - I_0*exp(-x).

        """
        functional = self

        class CTKLConvexConjGradient(Operator):
            # only defined when x<prior
            """The gradient operator of this functional."""

            def __init__(self):
                """Initialize a new instance."""
                super(CTKLConvexConjGradient, self).__init__(
                    functional.domain, functional.domain, linear=False)

            def _call(self, x):
                """Apply the gradient operator to the given point.

                """
                if functional.prior is None:
                    return np.log(functional.max_intens/(functional.max_intens - x))
                else:
                    return np.log(functional.max_intens/(functional.prior - x))

        return CTKLConvexConjGradient()

    @property
    def proximal(self):

        g = self.prior
        functional = self

        if g is not None and g not in self.space:
            raise TypeError('{} is not an element of {}'.format(g, functional.space))

        class ProximalCTKLConvexConj(Operator):

            """Proximal operator of the convex conjugate of the KL divergence."""

            def __init__(self, sigma):
                """Initialize a new instance.

                Parameters
                ----------
                sigma : positive float
                """
                super(ProximalCTKLConvexConj, self).__init__(
                    domain=functional.space, range=functional.space, linear=False)
                self.sigma = float(sigma)

            def _call(self, x, out):
                """Return ``self(x, out=out)``."""

                #if x is out:
                    # Handle aliased `x` and `out` (need original `x` later on)
                 #   x = x.copy()
                #else:
                #    out.assign(x)

                import scipy.special

                if g is None:
                    # If g is None, it is taken as I_0 times the one element
                    lambw = scipy.special.lambertw(
                        (functional.max_intens/self.sigma) * np.exp((functional.max_intens - x)/self.sigma))
                else:
                    lambw = scipy.special.lambertw(
                        (functional.max_intens/self.sigma) * np.exp((g - x)/self.sigma))

                if not np.issubsctype(self.domain.dtype, np.complexfloating):
                    lambw = lambw.real

                if g is None:
                    # If g is None, it is taken as I_0 times the one element
                    prox = functional.max_intens - self.sigma * lambw
                else:
                    prox = g - self.sigma * lambw

                prox = x.space.element(prox)
                out.assign(prox)
                #out = prox

        return ProximalCTKLConvexConj

    @property
    def convex_conj(self):
        """The convex conjugate functional of the KL-functional."""
        return CTKullbackLeibler(self.domain, self.prior, self.max_intens)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r})'.format(self.__class__.__name__,
                                       self.domain, self.prior)
