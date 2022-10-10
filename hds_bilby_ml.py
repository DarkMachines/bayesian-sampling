#!/usr/bin/env python
"""
All log-likelihoods for bilby runs
"""
import os
import bilby
import math
import numpy as np
import matplotlib.pyplot as plt

# To run GPU trained model with CPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

try:
    from tensorflow.keras.models import load_model
except ImportError:
    pass

# Output folders
label = 'rosenbrock'
outdir = 'outdir'

class MLFunction():
    """ Base class for functions that use a machine learning algorithm to
    provide function values """
    def __init__(self, *args, **kwargs):
        # Check if TF is installed
        try:
            load_model
        except NameError:
            raise ImportError(
                "The `tensorflow` package is not installed. This is needed in "
                "order to run `MLFunction`s. See the wiki on our GitHub "
                "project for installation instructions.")
        # Define object properties
        self.packageloc = self._get_package_location()
        self.model = None
        # Check definitions in parent class
        if not hasattr(self, 'modelname'):
            self.modelname = []
            raise Exception("MLFunction should define modelname.")

        is_standardised = hasattr(self, 'x_mean') and hasattr(self, 'x_stdev')
        if not is_standardised:
            self.x_mean, self.x_stdev = None, None
            raise Exception(
                "MLFunctions should either define x_mean and x_stdev or "
                "x_min and x_max.")

        is_standardised = hasattr(self, 'y_mean') and hasattr(self, 'y_stdev')
        if not is_standardised:
            self.y_mean, self.y_stdev = None, None
            raise Exception(
                "MLFunctions should either define y_mean and y_stdev or "
                "y_min and y_max.")

        super(MLFunction, self).__init__(*args, **kwargs)

    def _get_package_location(self):
        """ Get location in which the package was installed """
        this_dir, _ = os.path.split(__file__)
        return this_dir

    def _load_model(self):
        """ Load ML model from package """
        model_path = "ml_functions/{}/{}".format(self.modelname,
                                                    "model.hdf5")
        self.model = load_model(model_path)

    def _normalise(self, x, mu, sigma):
        return (x - mu) / sigma

    def _unnormalise(self, x, mu, sigma):
        return x * sigma + mu

    def _evaluate(self, x):
        # Load model is not already done
        if self.model is None:
            self._load_model()
        # Correct shape of x is needed
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        # Query to model
        x = self._normalise(x, self.x_mean, self.x_stdev)
        y = self.model.predict(x)
        y = self._unnormalise(y, self.y_mean, self.y_stdev)
        return y

class CMB(MLFunction, bilby.Likelihood):
    """
    N-dimensional Rosenbrock as defined in
    https://en.wikipedia.org/wiki/Rosenbrock_function

    Args:
        dimensionality: Number of input dimensions the function should take.

    """

    def __init__(self):
        self.modelname = 'lcdm'
        self.dim = 6
        self.x_mean = np.array([
            0.95985349, 1.04158987, 0.02235954, 0.11994063, 0.05296935, 
            3.06873361], np.float64)
        self.x_stdev = np.array([
            0.00836984, 0.00168727, 0.00043045, 0.00470686, 0.00899083, 
            0.02362278], np.float64)
            
        self.y_mean = 381.7929565376543
        self.y_stdev = 1133.7707883293974


        # Limit sampling between these hard borders, in order to ALWAYS remain
        # within the training box.
        ranges = []

        x_min = [0.92, 1.037, 0.02, 0.1, 0.0100002, 2.98
        ]
        x_max = [0.999999, 1.05, 0.024, 0.14, 0.097, 3.16
        ]

        for i in range(len(x_min)):
            ranges.append([x_min[i], x_max[i]])
        self.ranges = ranges

        parameters = ['omega_b', 'omega_cdm', 'theta_s', 'ln[A_s]', 'n_s', 'tau_r']
        labels = ['$\\omega_b$', '$\\omega_{cdm}$', '$\\theta_s$', '$ln[A_s]$', '$n_s$', '$\\tau_r$']

        #Uniform priors are assumed 
        priors = dict()
        i = 0
        for key in parameters:
          priors[key] = bilby.core.prior.Uniform(ranges[i][0], ranges[i][1], labels[i])
          i += 1

        self.priors = priors
        super(CMB, self).__init__(parameters=parameters)

    def log_likelihood(self):
        if self.model is None:
            self._load_model()
        x = np.array([self.parameters["x{0}".format(i)] for i in range(self.dim)])
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        x = self._normalise(x, self.x_mean, self.x_stdev)
        y = self.model.predict(x)
        y = self._unnormalise(y, self.y_mean, self.y_stdev)
        return -y

class MSSM7(MLFunction, bilby.Likelihood):
    """
    N-dimensional Rosenbrock as defined in
    https://en.wikipedia.org/wiki/Rosenbrock_function

    Args:
        dimensionality: Number of input dimensions the function should take.

    """

    def __init__(self):
        self.modelname = 'mssm7'
        self.dim = 12
        self.x_mean = np.array([
            -1.65550622e+02, 6.53242357e+07, -6.04267288e+06, 1.15227686e+07,
            -8.96546390e+02, 1.20880748e+03, 3.65456629e+01, 1.73423279e+02,
            1.18539912e-01, 4.00306869e-01, 4.31081695e+01, 5.80441328e+01
        ], np.float64)
        self.x_stdev = np.array([
            3.13242671e+03, 2.64037878e+07, 4.32735828e+06, 2.22328069e+07,
            2.33891832e+03, 6.20060930e+03, 1.40566829e+01, 3.83628710e-01,
            2.51362291e-04, 4.59609868e-02, 3.09244370e+00, 3.24780776e+00
        ], np.float64)
        self.y_mean = -262.5887645450105
        self.y_stdev = 7.461633956842537

        # Limit sampling between these hard borders, in order to ALWAYS remain
        # within the training box.
        ranges = []

        x_min = [
            -7.16775760e+03, 4.27547804e+05, -9.98192815e+07, -6.81824964e+07,
            -9.99995488e+03, -9.99999903e+03, 3.00597043e+00, 1.71060011e+02,
            1.16700013e-01, 2.00000156e-01, 1.90001455e+01, 3.10001673e+01
        ]
        x_max = [
            7.18253463e+03, 9.99999857e+07, 4.56142832e+05, 9.99999734e+07,
            9.99987623e+03, 9.99999881e+03, 6.99999394e+01, 1.75619963e+02,
            1.20299997e-01, 7.99999435e-01, 6.69997800e+01, 8.49983345e+01
        ]

        for i in range(len(x_min)):
            ranges.append([x_min[i], x_max[i]])
        self.ranges = ranges

        parameters = ['M2', 'mf2', 'mHu2', 'mHd2', 'Au', 'Ad', 'tanb', 'mt', 'alphas', 'rho0', 'sigmas', 'sigmal']
        labels = ['$M_2$', '$m_f^2$', '$m_{H_u}^2$', '$m_{H_d}^2$', '$A_u$', '$A_d$', 'tan \\beta', '$m_t$', '$\\alpha_s$', '$\\rho_0$', '$\\sigma_s$', '$\\sigma_l$']

        #Uniform priors are assumed
        priors = dict()
        i = 0
        for key in parameters:
          priors[key] = bilby.core.prior.Uniform(ranges[i][0], ranges[i][1], labels[i])
          i += 1

        self.priors = priors

        super(MSSM7, self).__init__(parameters=parameters)

    def log_likelihood(self):
        if self.model is None:
            self._load_model()
        x = np.array([self.parameters["x{0}".format(i)] for i in range(self.dim)])
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        x = self._normalise(x, self.x_mean, self.x_stdev)
        y = self.model.predict(x)
        y = self._unnormalise(y, self.y_mean, self.y_stdev)
        return y

class Rosenbrock(bilby.Likelihood):
    """
    N-dimensional Rosenbrock as defined in
    https://en.wikipedia.org/wiki/Rosenbrock_function
 
    We take a = 1, b = 100

    Args:
        dimensionality: Number of input dimensions the function should take.

    """

    def __init__(self, dimensionality=2):
        if dimensionality < 2:
            raise Exception("""Dimensionality of Rosenbrock function has to
                            be >=2.""")
        self.dim = dimensionality

        x_min = -3
        x_max = 3

        ranges = []
        for i in range(self.dim):
            ranges.append([x_min, x_max])
        self.ranges = ranges

        parameters = {"x{0}".format(i): 0 for i in range(self.dim)}

        #Uniform priors are assumed
        priors = bilby.core.prior.PriorDict()
        priors.update({"x{0}".format(i): bilby.core.prior.Uniform(ranges[i][0], ranges[i][1], "x{0}".format(i)) for i in range(dim)})
        
        self.priors = priors

        super(Rosenbrock, self).__init__(parameters=parameters)

    def log_likelihood(self):
        x = np.array([self.parameters["x{0}".format(i)] for i in range(self.dim)])
        y = 0
        for i in range(self.dim - 1):
            y += (100 * np.power(x[i + 1] - np.power(x[i], 2), 2) +
                  np.power(1 - x[i], 2))
        return -y

class Rastrigin(bilby.Likelihood):
    """
    N-dimensional Rastrigin function as defined in
    https://en.wikipedia.org/wiki/Rastrigin_function

    We take A = 100

    Args:
        dimensionality: Number of input dimensions the function should take.
    """

    def __init__(self, dimensionality=2):
        if dimensionality < 2:
            raise Exception("""Dimensionality of Rastrigin function has to
                            be >=2.""")
        self.a = 10
        self.dim = dimensionality

        x_min = -5.12
        x_max = 5.12

        ranges = []
        for i in range(self.dim):
            ranges.append([x_min, x_max])
        self.ranges = ranges

        parameters = {"x{0}".format(i): 0 for i in range(self.dim)}

        #Uniform priors are assumed
        priors = bilby.core.prior.PriorDict()
        priors.update({"x{0}".format(i): bilby.core.prior.Uniform(ranges[i][0], ranges[i][1], "x{0}".format(i)) for i in range(dim)})

        self.priors = priors

        super(Rastrigin, self).__init__(parameters=parameters)
                
    def log_likelihood(self):
        x = np.array([self.parameters["x{0}".format(i)] for i in range(self.dim)])             
        y = self.a * self.dim
        for i in range(self.dim):
            y += np.power(x[i], 2) - self.a * np.cos(2 * np.pi * x[i])

        return -y

class Himmelblau(bilby.Likelihood):
    """
    Himmelblau function as defined in
    https://en.wikipedia.org/wiki/Himmelblau%27s_function

    This is a 2-dimensional function with an application range bounded by -5
    and 5 for both input variables. 

    """

    def __init__(self):
        self.dim = 2

        x_min = -5
        x_max = 5

        ranges = []
        for i in range(self.dim):
            ranges.append([x_min, x_max])
        self.ranges = ranges

        parameters = {"x{0}".format(i): 0 for i in range(self.dim)}

        #Uniform priors are assumed
        priors = bilby.core.prior.PriorDict()
        priors.update({"x{0}".format(i): bilby.core.prior.Uniform(ranges[i][0], ranges[i][1], "x{0}".format(i)) for i in range(dim)})

        self.priors = priors

        super(Himmelblau, self).__init__(parameters=parameters)

    def log_likelihood(self):
        x = np.array([self.parameters["x{0}".format(i)] for i in range(self.dim)])
        y = np.power(np.power(x[0], 2) + x[1] - 11, 2) + np.power(x[0] + np.power(x[1], 2) - 7, 2)
        return -y
    
class EggBox(bilby.Likelihood):
    """
    N-dimensional EggBox as defined in arXiv:0809.3437.
    Testfunction as defined in arXiv:0809.3437

    Args:
        dimensionality: Number of input dimensions the function should take.

    """

    def __init__(self, dimensionality=2):
        if dimensionality < 2:
            raise Exception("""Dimensionality of EggBox function has to
                            be >=2.""")
        self.dim = dimensionality
        self.tmax = 5.0 * np.pi

        x_min = 0.
        x_max = 10.*np.pi

        ranges = []
        for i in range(self.dim):
            ranges.append([x_min, x_max])
        self.ranges = ranges

        parameters = {"x{0}".format(i): 0 for i in range(self.dim)}

        #Uniform priors are assumed
        priors = bilby.core.prior.PriorDict()
        priors.update({"x{0}".format(i): bilby.core.prior.Uniform(ranges[i][0], ranges[i][1], "x{0}".format(i)) for i in range(dim)})

        self.priors = priors

        super(EggBox, self).__init__(parameters=parameters)

    def log_likelihood(self):
        x = np.array([self.parameters["x{0}".format(i)] for i in range(self.dim)])
        y = 1

        for i in range(self.dim):
            y *= math.cos(x[i]/2)
        y = math.pow(2. + y, 5)
        return y

class GaussianShells(bilby.Likelihood):

    """
    N-dimensional GaussianShells as defined in arXiv:0809.3437.
    Both number of dimensions and number of rings are arbitrary but we assume that have 
    an equal radius and widths 

    Args:
        dimensionality: Number of input dimensions the function should take.
        modes: number of rings
        r: radius of rings
        w: width of rings
        c: is a n-rings x n-dims list with centers coordinates
    """

    def __init__(self, dimensionality=2, modes=2, r=2, w=0.1, c=[[-3.5, 0.],[3.5, 0.]]):
         if dimensionality < 2:
            raise Exception("""Dimensionality of GaussianShells function has to
                            be >=2.""")
         if modes != len(c):
            raise Exception("""Number of rings (modes) must be equal to number of centers (c""")   
         self.dim = dimensionality
         self.modes = modes
         self.r = r
         self.w = w
         self.c = np.array(c)
         self.const = math.log(1. / math.sqrt(2. * math.pi * w**2))  # normalization constant

         x_min = -6
         x_max = 6

         ranges = []
         for i in range(self.dim):
            ranges.append([x_min, x_max])
         self.ranges = ranges

         parameters = {"x{0}".format(i): 0 for i in range(self.dim)}

         #Uniform priors are assumed
         priors = bilby.core.prior.PriorDict()
         priors.update({"x{0}".format(i): bilby.core.prior.Uniform(ranges[i][0], ranges[i][1], "x{0}".format(i)) for i in range(dim)})

         self.priors = priors

         super(GaussianShells, self).__init__(parameters=parameters)


    # log-likelihood of a single shell
    def logcirc(self, theta, c):
         d = np.sqrt(np.sum((theta - c)**2, axis=-1))  # |theta - c|
         return self.const - (d - self.r)**2 / (2. * self.w**2)

    def log_likelihood(self):
     x = np.array([self.parameters["x{0}".format(i)] for i in range(self.dim)])
     y = -240.
     for i in range(self.modes):
      y = np.logaddexp(y, self.logcirc(x, self.c[i]))
     return y

#Functions are:
#6D-CMB, 12D-MSSM7, nD-Rosenbrock, nD-Rastrigin, 2D-Himmelblau, nD-EggBox, nD-GaussianShells

#Input is the number of dimensions for nD functions 
dim = 2
likelihood = Rosenbrock(dim)
priors = likelihood.priors

# And run sampler
result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynamic_dynesty', npoints=1000,
    outdir=outdir, label=label, bound='multi', sample='unif')

result.plot_corner()

plt.show()
