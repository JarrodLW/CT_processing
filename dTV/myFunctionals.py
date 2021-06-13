import numpy as np
import odl
import dTV.myOperators
from odl.solvers.functional.functional import Functional
from odl.solvers import (GroupL1Norm, L2NormSquared, L1Norm, Huber)
from odl.space.pspace import ProductSpace
from odl.operator.pspace_ops import BroadcastOperator
from odl.operator.operator import Operator
from odl.operator.default_ops import (IdentityOperator, 
                                      ConstantOperator, ZeroOperator)
from odl.operator.tensor_ops import PointwiseInner
import dTV.myDeform
from odl.operator import PointwiseNorm

from dTV.myAlgorithms import fgp_dual

##################################################
#################HELPER FUNCTIONS#################
##################################################


def total_variation(domain, grad): 
    L1 = GroupL1Norm(grad.range, exponent=2)
    return L1 * grad


def generate_vfield_from_sinfo(sinfo, grad, eta=1e-2):
    sinfo_grad = grad(sinfo)
    grad_space = grad.range
    norm = odl.PointwiseNorm(grad_space, 2)
    norm_sinfo_grad = norm(sinfo_grad)
    max_norm = np.max(norm_sinfo_grad)
    eta_scaled = eta * max(max_norm, 1e-4)
    norm_eta_sinfo_grad = np.sqrt(norm_sinfo_grad ** 2 +
                                  eta_scaled ** 2)
    xi = grad_space.element([g / norm_eta_sinfo_grad for g in sinfo_grad])

    return xi

    
def project_on_fixed_vfield(domain, vfield):

        class OrthProj(Operator):
            def __init__(self):
                super(OrthProj, self).__init__(domain, domain, linear=True)

            def _call(self, x, out):
                xi = vfield
                Id = IdentityOperator(domain)
                xiT = odl.PointwiseInner(domain, xi)
                xixiT = odl.BroadcastOperator(*[x*xiT for x in xi])
                gamma = 1
                P = (Id - gamma * xixiT)
                out.assign(P(x))

            @property
            def adjoint(self):
                return self

            @property
            def norm(self):
                return 1.

        return OrthProj()
    
    
#############################################
#################FUNCTIONALS#################
#############################################
    

def tv(domain, grad=None):
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
        grad.norm = 2 * np.sqrt(sum(1 / grad.domain.cell_sides**2))
    else:
        grad = grad

    f = odl.solvers.GroupL1Norm(grad.range, exponent=2)

    return f * grad


class TotalVariationBoxContraint(odl.solvers.Functional):
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

    def __init__(self, domain, alpha=1, grad=None, strong_convexity=0, 
                 lower=None, upper=None, prox_options={}, ):
        """
        """

        self.strong_convexity = strong_convexity

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
        self.tv = tv(domain, grad=grad)
        self.grad = self.tv.right
        self.constraint = odl.solvers.IndicatorBox(
                domain, lower=lower, upper=upper)
        self.l2 = 0.5 * odl.solvers.L2NormSquared(domain)
        self.proj_P = self.tv.left.convex_conj.proximal(1)
        self.proj_C = self.constraint.proximal(1)

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

        constraint = self.constraint(x)

        if constraint is np.inf:
            return np.inf
        else:
            out = self.alpha * self.tv(x)
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
#            return odl.IdentityOperator(self.domain)
            return self.proj_C

        else:
            def tv_prox(z, out=None):
                # solve
                # arg min_x 1/2 || S^(-1/2) (x - z) ||_2^2 + alpha ||G x||_1 + i_C(x)
                # via w = S^(-1/2) x, zz = 
                # S^(1/2) arg min_w 1/2 || w - S^(-1/2) z ||_2^2 
                #           + alpha ||G S^(1/2) w||_1 + i_{C S^(1/2)}(w)

                sgma = sigma 
                mu = self.strong_convexity
                
                if out is None:
                    out = z.space.zero()

                opts = self.prox_options

                if mu > 0:
                    sgma /= (1 + sgma * mu)
                    z /= (1 + sgma * mu)

                if opts['name'] == 'FGP':
                    if opts['warmstart']:
                        if opts['p'] is None:
                            opts['p'] = self.grad.range.zero()

                        p = opts['p']
                    else:
                        p = self.grad.range.zero()

                    out[:] = fgp_dual(p, z, sgma * self.alpha, opts['niter'], 
                                      self.grad, self.proj_C, self.proj_P, 
                                      tol=opts['tol'])

                    return out

                else:
                    raise NotImplementedError('Not yet implemented')

            return tv_prox
        
        
        
        
def directionalTotalVariationNonnegative(domain, alpha=1., sinfo=None, 
                 gamma=1, eta=1e-2, NonNeg=False, strong_convexity=0, prox_options={}):
                 
    if isinstance(domain, odl.ProductSpace):
        grad_basic = odl.Gradient(
                domain[0], method='forward', pad_mode='symmetric')
        
        pd = [odl.discr.diff_ops.PartialDerivative(
                domain[0], i, method='forward', pad_mode='symmetric') 
              for i in range(2)]
        cp = [odl.operator.ComponentProjection(domain, i) 
              for i in range(2)]
            
        if sinfo is None:
            grad = odl.BroadcastOperator(
                    *[pd[i] * cp[j]
                      for i in range(2) for j in range(2)])
            
        else:
            vfield = gamma * generate_vfield_from_sinfo(sinfo, grad_basic, eta)
            inner = odl.PointwiseInner(domain, vfield) * grad_basic
            grad = odl.BroadcastOperator(
                    *[pd[i] * cp[j] - vfield[i] * inner * cp[j] 
                      for i in range(2) for j in range(2)])
            grad.vfield = vfield
        
        grad.norm = grad.norm(estimate=True)
        
    else:
        grad_basic = odl.Gradient(
                domain, method='forward', pad_mode='symmetric')
        
        if sinfo is None:
            grad = grad_basic
        else:
            vfield = gamma * generate_vfield_from_sinfo(sinfo, grad_basic, eta)
            P = project_on_fixed_vfield(grad_basic.range, vfield)
            grad = P * grad_basic
            grad.vfield = vfield
            
        grad_norm = 2 * np.sqrt(sum(1 / grad_basic.domain.cell_sides**2))
        grad.norm = grad_norm
        
    if NonNeg:
        lower = 0
    else:
        lower = None

    return TotalVariationBoxContraint(domain, alpha=alpha, grad=grad,
                 strong_convexity=strong_convexity, lower=lower, upper=None, prox_options=prox_options)

   
class DataFitL2Disp(Functional):

    def __init__(self, space, data, forward=None):
        self.space = space
        self.image_space = self.space[0]
        self.affine_space = self.space[1]
        self.data = data
        if forward is None:
            self.forward = IdentityOperator(self.image_space)
        else:
            self.forward = forward
        
        self.datafit = 0.5 * L2NormSquared(data.space).translated(self.data)
        
        if isinstance(self.image_space, odl.ProductSpace):
            tangent_bundle = self.image_space[0].tangent_bundle
        else:
            tangent_bundle = self.image_space.tangent_bundle

        self.embedding = dTV.myOperators.Embedding_Affine(
                self.affine_space, tangent_bundle)
            
        super(DataFitL2Disp, self).__init__(space=space, linear=False,
                                            grad_lipschitz=np.nan)
    
    def __call__(self, x):
        xim = x[0]
        xaff = x[1]
        transl_operator = self.transl_op_fixed_im(xim)
        fctl = self.datafit * self.forward * transl_operator
        return fctl(xaff)
    
    
    def transl_op_fixed_im(self, im):
        
        if isinstance(self.image_space, odl.ProductSpace):
            deform_op = odl.BroadcastOperator(
                    dTV.myDeform.LinDeformFixedTempl(im[0]),
                    dTV.myDeform.LinDeformFixedTempl(im[1]))
        else:
            deform_op = dTV.myDeform.LinDeformFixedTempl(im)
            
        return deform_op * self.embedding
    
    
    def transl_op_fixed_vf(self, disp):
                
        # deform_op = myDeform.LinDeformFixedDisp(self.embedding(disp))
        deform_op = dTV.myDeform.LinDeformFixedDispAffine(self.embedding(disp), disp)
        
        if isinstance(self.image_space, odl.ProductSpace):
            deform_op = odl.DiagonalOperator(deform_op, len(self.image_space))

        return deform_op   
    
    
    def partial_gradient(self, i):
        if i == 0:
            functional = self
            class auxOperator(Operator):
                def __init__(self):
                    super(auxOperator, self).__init__(functional.space,
                                                      functional.image_space)
                def _call(self, x, out):
                    xim = x[0]
                    xaff = x[1]
                    transl_operator = functional.transl_op_fixed_vf(xaff)                  
                    func0 = functional.datafit * functional.forward * transl_operator
                    grad0 = func0.gradient                    
                    out.assign(grad0(xim))
                    
            return auxOperator()
        elif i == 1:
            functional = self
            class auxOperator(Operator):
                def __init__(self):
                    super(auxOperator, self).__init__(functional.space,
                                                      functional.affine_space)
                def _call(self, x, out):
                    xim = x[0]
                    xaff = x[1]                    
                    transl_operator = functional.transl_op_fixed_im(xim)                  
                    func1 = functional.datafit * functional.forward * transl_operator
                    grad1 = func1.gradient    
                    out.assign(grad1(xaff))                            
            return auxOperator()
        else:
            raise ValueError('No gradient defined for this variable')

    @property
    def gradient(self):
        return BroadcastOperator(*[self.partial_gradient(i) 
                                   for i in range(2)])



def cross_gradient(im_1, im_2):
    # takes two square numpy arrays, or equal sizes and computes the L2 normalised cross-gradient between the two

    height, width = im_1.shape # add assert equals statement; arrays should be same size

    domain = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=[height, width], dtype='float')
    im_1_odl = domain.element(im_1)
    im_2_odl = domain.element(im_2)

    grad = odl.Gradient(domain, method='forward', pad_mode='symmetric')
    grad_1 = grad(im_1_odl)
    grad_2 = grad(im_2_odl)

    pointwise_norm = PointwiseNorm(grad.range, 2)

    pointwise_norm_1 = pointwise_norm(grad_1)
    pointwise_norm_2 = pointwise_norm(grad_2)
    inner = odl.operator.tensor_ops.PointwiseInner(grad.range, grad_1)(grad_2)


    l2_norm = odl.solvers.functional.default_functionals.L2Norm(domain)

    cross_grad = l2_norm(pointwise_norm_1*pointwise_norm_2)**2 - l2_norm(inner)**2

    return cross_grad

