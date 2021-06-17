#!/usr/bin/env python
import datetime
import os
import pandas as pd
import random
import math
from functools import reduce
        
try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from bin.rational import RandomParameterization
from dreamcoder.domains.arithmetic.arithmeticPrimitives import (
    f0, f1, fpi, real_power, real_subtraction, real_addition,
    real_division, real_multiplication, real)
from dreamcoder.domains.list.listPrimitives import bootstrapTarget, bootstrapTarget_extra
from dreamcoder.dreamcoder import explorationCompression, commandlineArguments
from dreamcoder.grammar import Grammar
from dreamcoder.program import Program, Primitive
from dreamcoder.recognition import RecurrentFeatureExtractor, DummyFeatureExtractor
from dreamcoder.task import DifferentiableTask, squaredErrorLoss
from dreamcoder.type import tint, treal, tbool, baseType, tlist, arrow
from dreamcoder.utilities import eprint, numberOfCPUs, testTrainSplit

#tposint = baseType("posint")
treal = baseType("real")
#tzerotoone = baseType("zerotoone")

def makeTasksFromFile(name, request, data_transform, filename):
    data = pd.read_csv(filename)
    data = data[data.status.eq('Solve_Succeeded')]
    data['a2'] = data['a2'].clip(lower=0.0, upper=1.0)
    data['a1'] = data['a1'].clip(lower=0.0, upper=1.0)
    return DifferentiableTask(name,
                              request,
                              data_transform(data),
                              BIC=1.,
                              restarts=2,
                              steps=25,
                              maxParameters=5,
                              loss=squaredErrorLoss)

if __name__ == "__main__":

    test = []
    train = []

    # Task 1
    test.append(makeTasksFromFile("task1-test",
                                   arrow(treal, treal),
                                   lambda data : [((f1,), a1) for f1, a1 in data[['f1', 'a1']].to_numpy()],
                                   'prrttp/simulations/task1-test.csv'))
    train.append(makeTasksFromFile("task1-train",
                                    arrow(treal, treal),
                                    lambda data : [((f1,), a1) for f1, a1 in data[['f1', 'a1']].to_numpy()],
                                    'prrttp/simulations/task1-train.csv'))

    # Task 2
    test.append(makeTasksFromFile("task2-test",
                                   arrow(treal, treal, treal),
                                   lambda data : [((f1, f2), a1) for f1, f2, a1 in data[['f1', 'f2', 'a1']].to_numpy()],
                                   'prrttp/simulations/task2-test.csv'))
    train.append(makeTasksFromFile("task2-train",
                                    arrow(treal, treal, treal),
                                    lambda data : [((f1, f2), a1) for f1, f2, a1 in data[['f1', 'f2', 'a1']].to_numpy()],
                                    'prrttp/simulations/task2-train.csv'))

    # Task 3
    test.append(makeTasksFromFile("task3-test",
                                   arrow(treal, treal, treal, treal),
                                   lambda data : [((e12, f1, f2), a1) for e12, f1, f2, a1 in data[['e12', 'f1', 'f2', 'a1']].to_numpy()],
                                   'prrttp/simulations/task3-test.csv'))
    train.append(makeTasksFromFile("task3-train",
                                    arrow(treal, treal, treal, treal),
                                    lambda data : [((e12, f1, f2), a1) for e12, f1, f2, a1 in data[['e12', 'f1', 'f2', 'a1']].to_numpy()],
                                    'prrttp/simulations/task3-train.csv'))

    # Task 4
    test.append(makeTasksFromFile("task4-test",
                                   arrow(treal, treal, treal, treal),
                                   lambda data : [((ρ, f1, f2), a1) for ρ, f1, f2, a1 in data[['ρ', 'f1', 'f2', 'a1']].to_numpy()],
                                   'prrttp/simulations/task4-test.csv'))
    train.append(makeTasksFromFile("task4-train",
                                    arrow(treal, treal, treal, treal),
                                    lambda data : [((ρ, f1, f2), a1) for ρ, f1, f2, a1 in data[['ρ', 'f1', 'f2', 'a1']].to_numpy()],
                                    'prrttp/simulations/task4-train.csv'))

    # Task 5
    test.append(makeTasksFromFile("task5-test",
                                   arrow(treal, treal, treal, treal),
                                   lambda data : [((q, f1, f2), a1) for q, f1, f2, a1 in data[['q', 'f1', 'f2', 'a1']].to_numpy()],
                                   'prrttp/simulations/task5-test.csv'))
    train.append(makeTasksFromFile("task5-train",
                                    arrow(treal, treal, treal, treal),
                                    lambda data : [((q, f1, f2), a1) for q, f1, f2, a1 in data[['q', 'f1', 'f2', 'a1']].to_numpy()],
                                    'prrttp/simulations/task5-train.csv'))

    # Task 6
    test.append(makeTasksFromFile("task6-test",
                                   arrow(treal, treal, treal, treal, treal),
                                   lambda data : [((ρ, e12, f1, f2), a1) for ρ, e12, f1, f2, a1 in data[['ρ', 'e12', 'f1', 'f2', 'a1']].to_numpy()],
                                   'prrttp/simulations/task6-test.csv'))
    train.append(makeTasksFromFile("task6-train",
                                    arrow(treal, treal, treal, treal, treal),
                                    lambda data : [((ρ, e12, f1, f2), a1) for ρ, e12, f1, f2, a1 in data[['ρ', 'e12', 'f1', 'f2', 'a1']].to_numpy()],
                                    'prrttp/simulations/task6-train.csv'))

    # Task 7
    test.append(makeTasksFromFile("task7-test",
                                   arrow(treal, treal, treal, treal, treal),
                                   lambda data : [((q, e12, f1, f2), a1) for q, e12, f1, f2, a1 in data[['q', 'e12', 'f1', 'f2', 'a1']].to_numpy()],
                                   'prrttp/simulations/task7-test.csv'))
    train.append(makeTasksFromFile("task7-train",
                                    arrow(treal, treal, treal, treal, treal),
                                    lambda data : [((q, e12, f1, f2), a1) for q, e12, f1, f2, a1 in data[['q', 'e12', 'f1', 'f2', 'a1']].to_numpy()],
                                    'prrttp/simulations/task7-train.csv'))

    # Task 8
    test.append(makeTasksFromFile("task8-test",
                                   arrow(treal, treal, treal, treal, treal),
                                   lambda data : [((q, ρ, f1, f2), a1) for q, ρ, f1, f2, a1 in data[['q', 'ρ', 'f1', 'f2', 'a1']].to_numpy()],
                                   'prrttp/simulations/task8-test.csv'))
    train.append(makeTasksFromFile("task8-train",
                                    arrow(treal, treal, treal, treal, treal),
                                    lambda data : [((q, ρ, f1, f2), a1) for q, ρ, f1, f2, a1 in data[['q', 'ρ', 'f1', 'f2', 'a1']].to_numpy()],
                                    'prrttp/simulations/task8-train.csv'))

    # Task 9
    test.append(makeTasksFromFile("task9-test",
                                   arrow(treal, treal, treal, treal, treal, treal),
                                   lambda data : [((q, ρ, e12, f1, f2), a1) for q, ρ, e12, f1, f2, a1 in data[['q', 'ρ', 'e12', 'f1', 'f2', 'a1']].to_numpy()],
                                   'prrttp/simulations/task9-test.csv'))
    train.append(makeTasksFromFile("task9-train",
                                    arrow(treal, treal, treal, treal, treal, treal),
                                    lambda data : [((q, ρ, e12, f1, f2), a1) for q, ρ, e12, f1, f2, a1 in data[['q', 'ρ', 'e12', 'f1', 'f2', 'a1']].to_numpy()],
                                    'prrttp/simulations/task9-train.csv'))

    equationPrimitives = [
        # Possibly add individual ints
        #real,
        f0,
        f1,
        fpi,
        real_power,
        real_subtraction,
        real_addition,
        real_division,
        real_multiplication
        ]#+ [
         #  Program.parse(n)
         #  for n in [#"map","fold",
         #            #"empty","cons","car","cdr",
         #            #"zip",
         #            #"unfold", "range", "index", "length",
         #            "gt?.", "eq?.",
         #            "if", #"empty?",
         #            "true", "not",
         #            ]]
    baseGrammar = Grammar.uniform(equationPrimitives)
#    baseGrammar = Grammar.uniform(equationPrimitives + [p for p in bootstrapTarget_extra()])

    #eprint("Got %d equation discovery tasks..." % len(tasks))

    timestamp = datetime.datetime.now().isoformat()
    outputDirectory = "experimentOutputs/gameTheory/%s"%timestamp
    os.system("mkdir -p %s"%outputDirectory)

    explorationCompression(baseGrammar,
                           train,
                           outputPrefix="%s/gameTheory"%outputDirectory,
                           evaluationTimeout=0.1,
                           testingTasks=test,
                           **commandlineArguments(
                               compressor="ocaml",
                               featureExtractor=DummyFeatureExtractor,
                               iterations=10,
                               CPUs=numberOfCPUs(),
                               structurePenalty=0.001,
                               helmholtzRatio=0.5,
                               a=3,
                               maximumFrontier=10000,
                               topK=2,
                               pseudoCounts=10.0))
