import numpy as np

class SPSA(object):
    def __init__(self, function, initial_parameters, stepsize, bounds=None):
        self._function = function
        initial_parameters = np.array(initial_parameters)
        if bounds is None:
            self._bounds = None
        else:
            self._bounds = np.array(bounds)
        if not self._bounds is None:
            if not (self._bounds.shape[0] == initial_parameters.shape[0]):
                raise ValueError("different number of bounds and parameters")
            for j in range(len(initial_parameters)):
                if (initial_parameters[j] >= self.bounds[j, 0]) and (initial_parameters[j] <= self.bounds[j, 1]):
                    continue
                raise ValueError("an initial parameter is out of bounds")

        self._initial_parameters = self._bounded_to_infinite(initial_parameters)

        self._calls = 0
        self._parameters = np.array(self._initial_parameters)
        self._parameter_values = []
        self._function_values = []

    @property
    def calls(self):
        return self._calls

    @property
    def initial_parameters(self):
        return _infinite_to_bounded(self._initial_parameters)

    @property
    def parameters(self):
        return self._infinite_to_bounded(self._parameters)

    @property
    def bounds(self):
        if self._bounds is None:
            return None
        return np.array(self._bounds)

    @property
    def parameter_values(self):
        return list(self._parameter_values)

    @property
    def function_values(self):
        return list(self._function_values)

    def _infinite_to_bounded(self, params):
        if self._bounds is None:
            return np.array(params)
        transformed = np.empty_like(params)
        for j, param in enumerate(params):
            boundmid = (self.bounds[j,1] + self.bounds[j,0])/2
            bounddel = (self.bounds[j,1] - self.bounds[j,0])/2
            transformed[j] = (bounddel*(2/np.pi)*np.arctan(param))+boundmid
        return transformed

    def _bounded_to_infinite(self, params):
        if self._bounds is None:
            return np.array(params)
        transformed = np.empty_like(params)
        for j, param in enumerate(params):
            boundmid = (self.bounds[j,1] + self.bounds[j,0])/2
            bounddel = (self.bounds[j,1] - self.bounds[j,0])/2
            transformed[j] = np.tan( ((param - boundmid)/bounddel)*(np.pi/2) )
        return transformed

    def set_parameters(self, new_parameters):
        if not len(new_parameters) == len(self._parameters):
            raise ValueError
        new_parameters = self._bounded_to_infinite(new_parameters)
        self._parameters = new_parameters
        return self

    def _evaluate_function(self, params=None):
        if params is None:
            params = self._parameters
        params = self._infinite_to_bounded(params)
        output = self._function(*params)
        self._calls += 1
        self._parameter_values.append(params)
        self._function_values.append(output)
        return output

    def _grad(self, stepsize, params=None, function_vals=False):
        if params is None:
            params = self._parameters
        delta = (2.0*np.random.default_rng().uniform(size=len(params)))-1.0
        delta = np.sign(delta)
        params_plus = params+(stepsize*delta)
        params_minus = params-(stepsize*delta)
        func_plus = self._evaluate_function(params=params_plus)
        func_minus = self._evaluate_function(params=params_minus)
        grad_est = (func_plus - func_minus)/(2*stepsize*delta)
        if function_vals:
            return grad_est, func_plus, func_minus
        else:
            return grad_est

    def update(self, gradient_stepsize, stepsize):
        params = self._parameters
        grad = self._grad(gradient_stepsize, params=params)
        params = params - (stepsize*grad)
        self._parameters = params
        return self

    def optimize(self, iterations, verbose=False):
        initial_iterations = int(np.round(0.1*iterations))
        if initial_iterations % 2 == 1:
            initial_iterations += 1

        A = int(np.round(0.1*(iterations-initial_iterations)))
        alpha = 0.602
        gamma = 0.101
        c0 = 1

        fps, fms = [], []
        grad, fp, fm = self._grad(c0, params=self._parameters, function_vals=True)
        grad = np.abs(grad)
        fps.append(fp)
        fms.append(fm)
        for j in range(1, initial_iterations):
            newgrad, newfp, newfm = self._grad(0.01, params=self._parameters, function_vals=True)
            grad = grad + np.abs(newgrad)
            fps.append(newfp)
            fms.append(newfm)
        grad = grad/initial_iterations
        c = (np.std(fps) + np.std(fms))/2
        tmp = np.max(self.bounds[:,1] - self.bounds[:,0])/(0.01*(iterations-initial_iterations))
        a = tmp*np.max(((A+1)**alpha)/grad)

        for j in range(iterations - initial_iterations):
            if verbose:
                print(f"Iteration {initial_iterations+j+1} of {iterations}", end="\r")
            aj = a/((A+j+2)**alpha)
            cj = c/((j+2)**gamma)

            self.update(cj, aj)
