# Copyright 2015 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import numpy as np
import logging
from SparseArray import SparseArray
from .node import Variable
from .node import Add, Mul, Div, Fabs, Exp, Sqrt, Sin, Cos, Log1p
from .node import Sq, Min, Max
from .node import Atan2, Hypot, Acos, Asin, Atan, Tan, Cosh, Sinh
from .node import Tanh, Acosh, Asinh, Atanh, Expm1, Log, Log2, Log10
from .node import Lgamma, Sign, Ceil, Floor, NaiveBayes, NaiveBayesMN
from .model import Model
from .population import SteadyState
from .utils import tonparray
from .function_selection import FunctionSelection
from .naive_bayes import NaiveBayes as NB
from .bagging_fitness import BaggingFitness
import time
import importlib
import inspect


class EvoDAG(object):
    def __init__(self, generations=np.inf, popsize=10000,
                 seed=0, tournament_size=2,
                 early_stopping_rounds=-1,
                 function_set=[Add, Mul, Div, Fabs,
                               Exp, Sqrt, Sin, Cos, Log1p,
                               Sq, Min, Max, Atan2, Hypot, Acos, Asin, Atan,
                               Tan, Cosh, Sinh, Tanh, Acosh, Asinh, Atanh,
                               Expm1, Log, Log2, Log10, Lgamma, Sign,
                               Ceil, Floor, NaiveBayes, NaiveBayesMN],
                 tr_fraction=0.5, population_class=SteadyState,
                 number_tries_feasible_ind=30, time_limit=None,
                 unique_individuals=True, classifier=True,
                 labels=None, all_inputs=False, random_generations=0, fitness_function='BER',
                 min_density=0.8, multiple_outputs=False, function_selection=True,
                 fs_tournament_size=2, finite=True, pr_variable=0.33,
                 share_inputs=False, input_functions=None, **kwargs):
        self._fitness_function = fitness_function
        self._bagging_fitness = BaggingFitness(base=self)
        generations = np.inf if generations is None else generations
        self._pr_variable = pr_variable
        self._share_inputs = share_inputs
        self._finite = finite
        self._generations = generations
        self._popsize = popsize
        self._classifier = classifier
        self._number_tries_feasible_ind = number_tries_feasible_ind
        self._unfeasible_counter = 0
        self._number_tries_unique_args = 3
        self._tr_fraction = tr_fraction
        if early_stopping_rounds is not None and early_stopping_rounds < 0:
            early_stopping_rounds = popsize
        self._early_stopping_rounds = early_stopping_rounds
        self._tournament_size = tournament_size
        self._seed = seed
        self._labels = labels
        self._multiclass = False
        self._function_set = function_set
        self._function_selection = function_selection
        self._fs_tournament_size = fs_tournament_size
        density_safe = [k for k, v in enumerate(function_set) if v.density_safe]
        self._function_selection_ins = FunctionSelection(nfunctions=len(self._function_set),
                                                         seed=seed,
                                                         tournament_size=self._fs_tournament_size,
                                                         nargs=map(lambda x: x.nargs,
                                                                   function_set),
                                                         density_safe=density_safe)
        self._min_density = min_density
        self._function_selection_ins.min_density = self._min_density
        self._time_limit = time_limit
        self._init_time = time.time()
        self._random_generations = random_generations
        if not inspect.isclass(population_class):
            pop = importlib.import_module('EvoDAG.population')
            population_class = getattr(pop, population_class)
        self._set_input_functions(input_functions)
        self._population_class = population_class
        np.random.seed(self._seed)
        self._unique_individuals = unique_individuals
        self._unique_individuals_set = set()
        self._logger = logging.getLogger('EvoDAG')
        self._all_inputs = all_inputs
        if self._generations == np.inf and tr_fraction == 1:
            raise RuntimeError("Infinite evolution, set generations\
            or tr_fraction < 1 ")
        self._multiple_outputs = multiple_outputs
        self._extras = kwargs

    def _set_input_functions(self, input_functions):
        if input_functions is not None:
            if not isinstance(input_functions, list):
                input_functions = [input_functions]
            r = []
            for f in input_functions:
                if not inspect.isclass(f):
                    _ = importlib.import_module('EvoDAG.node')
                    f = getattr(_, f)
                    r.append(f)
                else:
                    r.append(f)
            self._input_functions = r
        else:
            self._input_functions = input_functions
        
    def get_params(self):
        "Parameters used to initialize the class"
        import inspect
        a = inspect.getargspec(self.__init__)[0]
        out = dict()
        for key in a[1:]:
            value = getattr(self, "_%s" % key, None)
            out[key] = value
        return out

    def clone(self):
        "Clone the class without the population"
        return self.__class__(**self.get_params())

    @property
    def signature(self):
        "Instance file name"
        kw = self.get_params()
        keys = sorted(kw.keys())
        l = []
        for k in keys:
            n = k[0] + k[-1]
            v = kw[k]
            if k == 'function_set':
                v = "_".join([x.__name__[0] +
                              x.__name__[-1] +
                              str(x.nargs) for x in kw[k]])
            elif k == 'population_class':
                v = kw[k].__name__
            else:
                v = str(v)
            l.append('{0}_{1}'.format(n, v))
        return '-'.join(l)

    @property
    def popsize(self):
        """Population size"""
        return self._popsize

    @property
    def generations(self):
        """Number of generations"""
        return self._generations

    @property
    def X(self):
        """Features or variables used in the training and validation set"""
        return self._X

    @X.setter
    def X(self, v):
        self._X = Model.convert_features(v)
        self.nvar = len(self._X)

    @property
    def Xtest(self):
        "Features or variables used in the test set"
        return [x.hy_test for x in self.X]

    @Xtest.setter
    def Xtest(self, v):
        Model.convert_features_test_set(self.X, v)

    @property
    def y(self):
        """Dependent variable"""
        return self._y

    @y.setter
    def y(self, v):
        if isinstance(v, np.ndarray):
            v = SparseArray.fromlist(v)
        if self._classifier:
            self._multiple_outputs = True
            if self._labels is None:
                self.nclasses(v)
            return self._bagging_fitness.multiple_outputs_cl(v)
        elif self._multiple_outputs:
            return self._bagging_fitness.multiple_outputs_regression(v)
        elif self._tr_fraction < 1:
            for i in range(self._number_tries_feasible_ind):
                self._bagging_fitness.set_regression_mask(v)
                flag = self._bagging_fitness.test_regression_mask(v)
                if flag:
                    break
            if not flag:
                msg = "Unsuitable validation set (RSE: average equals zero)"
                raise RuntimeError(msg)
        else:
            self._mask = 1.0
        self._ytr = v * self._mask
        self._y = v

    @property
    def function_set(self):
        "List containing the functions used to create the individuals"
        return self._function_set

    @property
    def nvar(self):
        """Number of features or variables"""
        return self._nvar

    @nvar.setter
    def nvar(self, v):
        self._nvar = v

    @property
    def naive_bayes(self):
        try:
            return self._naive_bayes
        except AttributeError:
            if hasattr(self, '_y_klass'):
                self._naive_bayes = NB(mask=self._mask_ts, klass=self._y_klass,
                                       nclass=self._labels.shape[0])
            else:
                self._naive_bayes = None
        return self._naive_bayes

    @property
    def population(self):
        "Class containing the population and all the individuals generated"
        try:
            return self._p
        except AttributeError:
            self._p = self._population_class(base=self,
                                             tournament_size=self._tournament_size,
                                             classifier=self._classifier,
                                             labels=self._labels,
                                             es_extra_test=self.es_extra_test,
                                             popsize=self._popsize,
                                             random_generations=self._random_generations)
            return self._p

    @property
    def init_popsize(self):
        if self._all_inputs:
            return self._init_popsize
        return self.popsize

    def es_extra_test(self, v):
        """This function is called from population before setting
        the early stopping individual and after the comparisons with
        the validation set fitness"""
        return True

    def _random_leaf(self, var):
        v = Variable(var, ytr=self._ytr, finite=self._finite, mask=self._mask,
                     naive_bayes=self.naive_bayes)
        if not v.eval(self.X):
            return None
        if not v.isfinite():
            return None
        if not self._bagging_fitness.set_fitness(v):
            return None
        return v

    def random_leaf(self):
        "Returns a random variable with the associated weight"
        for i in range(self._number_tries_feasible_ind):
            var = np.random.randint(self.nvar)
            v = self._random_leaf(var)
            if v is None:
                continue
            return v
        raise RuntimeError("Could not find a suitable random leaf")

    def unique_individual(self, v):
        "Test whether v has not been explored during the evolution"
        return v not in self._unique_individuals_set

    def unfeasible_offspring(self):
        self._unfeasible_counter += 1
        return None

    def _random_offspring(self, func, args):
        f = func(args, ytr=self._ytr, naive_bayes=self.naive_bayes,
                 finite=self._finite, mask=self._mask)
        if self._unique_individuals:
            sig = f.signature()
            if self.unique_individual(sig):
                self._unique_individuals_set.add(sig)
            else:
                return self.unfeasible_offspring()
        f.height = max([self.population.hist[x].height for x in args]) + 1
        if not f.eval(self.population.hist):
            return self.unfeasible_offspring()
        if not f.isfinite():
            return self.unfeasible_offspring()
        if not self._bagging_fitness.set_fitness(f):
            return self.unfeasible_offspring()
        return f

    def get_unique_args(self, func):
        args = {}
        res = []
        p_tournament = self.population.tournament
        n_tries = self._number_tries_unique_args
        for j in range(func.nargs):
            k = p_tournament()
            for _ in range(n_tries):
                if k not in args:
                    args[k] = 1
                    res.append(k)
                    break
                else:
                    k = p_tournament()
        if len(res) < func.min_nargs:
            return None
        return res

    def get_sample_population(self,size):
        if size >= self.population.popsize or size == 0:
            size = self.population.popsize
        if size==0:
            size = 1
        res = []
        done = {}
        for _ in range(size): 
            k = np.random.randint(self.population.popsize)
            while k in done:
                k = np.random.randint(self.population.popsize)
            done[k] = 1
            res.append(k)
        return res,size

    @staticmethod
    def calculate_desired(func,y,hy,unique=True):
        if isinstance(y,list):
            desired = []
            desired_unique = []
            for i in range(len(y)):
                if func.symbol == '+':
                    desired.append(SparseArray.sub(y[i],hy[i]))
                elif func.symbol == '*':
                    desired.append(SparseArray.div(y[i],hy[i]))
                elif func.symbol == '/':
                    desired.append(SparseArray.div(hy[i],y[i]))
                desired_unique.append(SparseArray.unit_vector(desired[i]))
        else:
            if func.symbol == '+':
                desired = SparseArray.sub(y,hy)
            elif func.symbol == '*':
                desired = SparseArray.div(y,hy)
            elif func.symbol == '/':
                desired = SparseArray.div(hy,y)
            desired_unique = SparseArray.unit_vector(desired)
        if unique:
            return desired_unique
        else:
            return desired

    ''''''
    @staticmethod
    def calculate_semantic_difference(semantics1,semantics2):
        dif = 0
        if isinstance(semantics1,list):
            for i in range(len(semantics1)):
                dif += sum(SparseArray.fabs(SparseArray.sub(semantics1[i],semantics2[i])).data)
        else:
            dif = sum(SparseArray.fabs(SparseArray.sub(semantics1,semantics2)).data) 
        return dif
    ''''''
    def cosine_similarity(semantics1,semantics2):
        sim = 0
        if isinstance(semantics1,list):
            for i in range(len(semantics1)):
                prod = SparseArray.dot(semantics1[i],semantics2[i])
                n1 = math.sqrt(sum(SparseArray.sq(semantics1[i]).data))
                n2 = math.sqrt(sum(SparseArray.sq(semantics2[i]).data))
                if n1 == 0 or n2 == 0 or prod == 0:
                    cossim = 0
                else:
                    cossim = prod/(n1*n2)
                sim += cossim
        else:
            prod = SparseArray.dot(semantics1,semantics2)
            n1 = math.sqrt(sum(SparseArray.sq(semantics1).data))
            n2 = math.sqrt(sum(SparseArray.sq(semantics2).data))
            if n1 == 0 or n2 == 0 or prod == 0:
                sim = 0
            else:
                sim = prod/(n1*n2)
        return sim
    ''''''
    def tournament_desired(self,desired_semantics,size,args,unique=True):
        sample,size = self.get_sample_population(size)
        Dif = np.zeros((size,2),float)
        for i in range(size):
            k = sample[i]
            Dif[i,0] = k
            if unique:
                Dif[i,1] = EvoDAG.calculate_semantic_difference(desired_semantics,self.population.hist[self.population.population[k].position].hy_unique)
            else:
                Dif[i,1] = EvoDAG.calculate_semantic_difference(desired_semantics,self.population.hist[self.population.population[k].position].hy)
        arguments = Dif[ np.argsort(Dif[:,1]),0]
        for arg in arguments:
            if arg not in args:
                return int(arg)
        return 0
    ''''''  
    '''  
    def tournament_desired(self,desired_semantics,size,args):
        sample,size = self.get_sample_population(size)
        Sim = np.zeros((size,2),float)
        for i in range(size):
            k = sample[i]
            Sim[i,0] = k
            Sim[i,1] = EvoDAG.cosine_similarity(desired_semantics,self.population.hist[self.population.population[k].position].hy)
        arguments = Sim[ np.argsort(Sim[:,1]),0 ]
        for arg in reversed(arguments):
            if arg not in args:
                return int(arg)
        return 0
    '''
    def tournament_closer(self,func,size):
        if random.random() <= 0.5:
            return self.population.tournament()

        m = self.population.tournament()
        args = [m]
        argsi = [self.population.population[x].position for x in args]
        individualm = self._random_offspring(func,argsi)
        fitnessm = individualm.fitness if individualm is not None else -10000

        n = self.population.tournament()
        args = [n]
        argsi = [self.population.population[x].position for x in args]
        individualn = self._random_offspring(func,argsi)
        fitnessn = individualm.fitness if individualm is not None else -10000
        
        if(fitnessm >= fitnessn):
            return m
        else: 
            return n

        return arg

    ''''''
    @staticmethod
    def calculate_orthogonality(vectors,vector):
        o = 0
        for v in vectors:
            o+= abs(SparseArray.dot(v,vector))
        return o
    ''''''
    def tournament_orthogonality(self,size,args):
        sample,size = self.get_sample_population(size)
        vectors = []
        for k in args:
            if isinstance( self.population.hist[self.population.population[k].position].hy,list ):
                vectors.append( self.population.hist[self.population.population[k].position].hy[0] )
            else:
                vectors.append( self.population.hist[self.population.population[k].position].hy )
        
        Dif = np.zeros((size,2),float)
        for i in range(size):
            k = sample[i]
            vector = self.population.hist[self.population.population[k].position].hy
            if isinstance( self.population.hist[self.population.population[k].position].hy,list ):
                vector = self.population.hist[self.population.population[k].position].hy[0]
            Dif[i,0] = k
            Dif[i,1] = EvoDAG.calculate_orthogonality(vectors,vector)
            #Dif[i,1] = 0
            #for j in range(len(vectors)):
            #    Dif[i,1] += EvoDAG.cosine_similarity(vectors[j],vector)
        arguments = Dif[ np.argsort(Dif[:,1]),0]
        for arg in arguments:
            if arg not in args:
                return int(arg)
        return 0

    def get_args(self, func):
        args = []

        ''''''
        if func.nargs == 1:
            k = self.tournament_closer(func,2)
            args.append(k)
            return args
        ''''''
        #Searching n arguments based on orthogonality
        ''''''
        if func.symbol == '+' or func.symbol == 'NB' or func.symbol == 'MN':
            k = self.population.tournament()
            args.append(k)
            while len(args)<func.nargs:
                m = self.tournament_orthogonality(2,args)
                args.append(m)
            return args
        ''''''
        ''''''
        #Searching n arguments based on desired unique vectors
        if func.symbol == '*' or func.symbol == '/':
            k = self.population.tournament()
            args.append(k)
            desired_semantics = EvoDAG.calculate_desired(func,self.y,self.population.hist[self.population.population[k].position].hy,unique=True)
            #j = self.tournament_desired(desired_unique,2,args)
            j = self.tournament_desired(desired_semantics,2,args,unique=True)
            args.append(j)

            while len(args)<func.nargs:
                argsi = [self.population.population[x].position for x in args]
                individual = self._random_offspring(func, argsi)
                if individual is None:
                    break
                desired_semantics = EvoDAG.calculate_desired(func,self.y,individual.hy,unique=True)
                #m = self.tournament_desired(desired_unique,2,args)
                m = self.tournament_desired(desired_semantics,2,args,unique=True)
                args.append(m)
            return args
        ''''''

        if func.unique_args:
            return self.get_unique_args(func)
        for j in range(func.nargs):
            k = self.population.tournament()
            for _ in range(self._number_tries_unique_args):
                if k not in args:
                    break
                else:
                    k = self.population.tournament()
            args.append(k)
        return args

    def random_offspring(self):
        "Returns an offspring with the associated weight(s)"
        function_set = self.function_set
        function_selection = self._function_selection_ins
        function_selection.density = self.population.density
        function_selection.unfeasible_functions.clear()
        for i in range(self._number_tries_feasible_ind):
            if self._function_selection:
                func_index = function_selection.tournament()
            else:
                func_index = function_selection.random_function()
            func = function_set[func_index]
            args = self.get_args(func)
            if args is None:
                continue
            args = [self.population.population[x].position for x in args]
            f = self._random_offspring(func, args)
            if f is None:
                function_selection.unfeasible_functions.add(func_index)
                continue
            function_selection[func_index] = f.fitness
            return f
        raise RuntimeError("Could not find a suitable random offpsring")

    def create_population(self):
        "Create the initial population"
        self.population.create_population()

    def stopping_criteria(self):
        "Test whether the stopping criteria has been achieved."
        if self._time_limit is not None:
            if time.time() - self._init_time > self._time_limit:
                return True
        if self.generations < np.inf:
            inds = self.popsize * self.generations
            flag = inds <= len(self.population.hist)
        else:
            flag = False
        if flag:
            return True
        est = self.population.estopping
        if self._tr_fraction < 1:
            if est is not None and est.fitness_vs == 0:
                return True
        esr = self._early_stopping_rounds
        if self._tr_fraction < 1 and esr is not None and est is not None:
            position = self.population.estopping.position
            if position < self.init_popsize:
                position = self.init_popsize
            return (len(self.population.hist) +
                    self._unfeasible_counter -
                    position) > esr
        return flag

    def nclasses(self, v):
        "Number of classes of v, also sets the labes"
        if not self._classifier:
            return 0
        if isinstance(v, list):
            self._labels = np.arange(len(v))
            return
        if not isinstance(v, np.ndarray):
            v = tonparray(v)
        self._labels = np.unique(v)
        return self._labels.shape[0]

    def replace(self, a):
        "Replace an individual in the population with individual a"
        self.population.replace(a)

    def fit(self, X, y, test_set=None):
        "Evolutive process"
        self._init_time = time.time()
        self.X = X
        nclasses = self.nclasses(y)
        if self._classifier and self._multiple_outputs:
            pass
        elif nclasses > 2:
            assert False
            self._multiclass = True
            return self.multiclass(X, y, test_set=test_set)
        self.y = y
        if test_set is not None:
            self.Xtest = test_set
        for _ in range(self._number_tries_feasible_ind):
            self._logger.info("Starting evolution")
            try:
                self.create_population()
            except RuntimeError as err:
                self._logger.info("Done evolution (RuntimeError (%s), hist: %s)" % (err, len(self.population.hist)))
                return self
            self._logger.info("Population created (hist: %s)" % len(self.population.hist))
            if len(self.population.hist) >= self._tournament_size:
                break
        if len(self.population.hist) < self._tournament_size:
            self._logger.info("Done evolution (hist: %s)" % len(self.population.hist))
            return self
        while not self.stopping_criteria():
            try:
                a = self.random_offspring()
            except RuntimeError as err:
                self._logger.info("Done evolution (RuntimeError (%s), hist: %s)" % (err, len(self.population.hist)))
                return self
            self.replace(a)
        self._logger.info("Done evolution (hist: %s)" % len(self.population.hist))
        return self

    def trace(self, n):
        "Restore the position in the history of individual v's nodes"
        return self.population.trace(n)

    def model(self, v=None):
        "Returns the model of node v"
        return self.population.model(v=v)

    def decision_function(self, v=None, X=None):
        "Decision function i.e. the raw data of the prediction"
        m = self.model(v=v)
        return m.decision_function(X)

    def predict(self, v=None, X=None):
        """In classification this returns the classes, in
        regression it is equivalent to the decision function"""
        m = self.model(v=v)
        return m.predict(X)

RGP = EvoDAG
RootGP = RGP
