'''
import random
import numpy as np
import math
import logging
from .base import EvoDAG
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

def get_sample_population(evodag,size):
    if size >= evodag.population.popsize or size == 0:
        size = evodag.population.popsize
    if size==0:
        size = 1
    res = []
    done = {}
    for _ in range(size): 
        k = np.random.randint(evodag.population.popsize)
        while k in done:
            k = np.random.randint(evodag.population.popsize)
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
def tournament_desired(evodag,desired_semantics,size,args,unique=True):
    sample,size = evodag.get_sample_population(size)
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
''''''
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
''''''
def tournament_closer(evodag,func,size):
    if random.random() <= 0.5:
        return evodag.population.tournament()

    m = evodag.population.tournament()
    args = [m]
    argsi = [evodag.population.population[x].position for x in args]
    individualm = evodag._random_offspring(func,argsi)
    fitnessm = individualm.fitness if individualm is not None else -10000

    n = evodag.population.tournament()
    args = [n]
    argsi = [evodag.population.population[x].position for x in args]
    individualn = evodag._random_offspring(func,argsi)
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
def tournament_orthogonality(evodag,size,args):
    sample,size = evodag.get_sample_population(size)
    vectors = []
    for k in args:
        if isinstance( evodag.population.hist[evodag.population.population[k].position].hy,list ):
            vectors.append( evodag.population.hist[evodag.population.population[k].position].hy[0] )
        else:
            vectors.append( evodag.population.hist[evodag.population.population[k].position].hy )
    
    Dif = np.zeros((size,2),float)
    for i in range(size):
        k = sample[i]
        vector = evodag.population.hist[evodag.population.population[k].position].hy
        if isinstance( evodag.population.hist[evodag.population.population[k].position].hy,list ):
            vector = evodag.population.hist[evodag.population.population[k].position].hy[0]
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
    '''
