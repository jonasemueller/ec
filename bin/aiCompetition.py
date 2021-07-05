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

from dreamcoder.domains.arithmetic.arithmeticPrimitives import (
    f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, real_power, real_subtraction, real_addition,
    real_division, real_multiplication, real)
#from dreamcoder.domains.list.listPrimitives import bootstrapTarget, bootstrapTarget_extra
from dreamcoder.dreamcoder import explorationCompression, commandlineArguments
from dreamcoder.grammar import Grammar
from dreamcoder.program import Program, Primitive, Invented
from dreamcoder.recognition import RecurrentFeatureExtractor, DummyFeatureExtractor, ImageFeatureExtractor
from dreamcoder.task import DifferentiableTask, squaredErrorLoss
from dreamcoder.type import tint, treal, tbool, baseType, tlist, arrow
from dreamcoder.utilities import eprint, numberOfCPUs

def conditionOn(data, parameters):
    """Conditioning the data on certain parameter values makes the data set equal
    to one from a simpler model."""
    for parameter in parameters:
        if parameter == 'ρ':
            data = data[data.ρ.eq(5000.)]
        elif parameter == 'q':
            data = data[data.q.eq(1.)]
    return data

def loadTaskDataFromFile(filename):
    data = pd.read_csv(filename)
    data = data[data.status.eq('Solve_Succeeded')]
    data['a2'] = data['a2'].clip(lower=0.0, upper=1.0)
    data['a1'] = data['a1'].clip(lower=0.0, upper=1.0)
    return data

def makeTask(name, request, taskData, actualParameters):
    return DifferentiableTask(name,
                              request,
                              taskData,
                              BIC=1.,
                              temperature=0.1,
                              restarts=300,
                              steps=50,
                              actualParameters=actualParameters,
                              maxParameters=4,
                              loss=squaredErrorLoss)

def randomCoefficient(m=5):
    t = 0.3
    f = t + (random.random() * (m - t))
    if random.random() > 0.5:
        f = -f
    f = float("%0.1f" % f)
    return f

class RandomParameterization(object):
    def primitive(self, e):
        if e.name == 'REAL':
            return Primitive(str(e), e.tp, randomCoefficient())
        return e

    def invented(self, e): return e.body.visit(self)

    def abstraction(self, e): return Abstraction(e.body.visit(self))

    def application(self, e):
        return Application(e.f.visit(self), e.x.visit(self))

    def index(self, e): return e

RandomParameterization.single = RandomParameterization()

class FeatureExtractor(ImageFeatureExtractor):
    special = 'differentiable'

    def __init__(self, tasks, testingTasks=[], cuda=False, H=64):
        self.recomputeTasks = True
        super(FeatureExtractor, self).__init__(inputImageDimension=64,
                                               channels=1)
        self.tasks = tasks

    def featuresOfTask(self, t):
        return self(t.features)

    def taskOfProgram(self, p, t):
        p = p.visit(RandomParameterization.single)

        def f(x): return p.runWithArguments([x])
        t = makeTask(str(p), f, None)
        if t is None:
            return None
        t.features = drawFunction(200, 5., t.f)
        delattr(t, 'f')
        return t

def groundTruthSolutions():
    solutions = ["(lambda (e f2 f1) (/. (-. f1 f2) e))"]
    return [Invented(Program.parseHumanReadable(s)) for s in solutions]

if __name__ == "__main__":

    test = []
    train = []

    # task 1
    taskData = loadTaskDataFromFile('data/aiCompetition/aiCompetitionTasks/task1-train.csv')
    taskData = [((f1,), a1) for f1, a1 in taskData[['f1', 'a1']].to_numpy()]
    train.append(makeTask("task1-train", arrow(treal, treal), taskData, 1))

    taskData = loadTaskDataFromFile('data/aiCompetition/aiCompetitionTasks/task1-test.csv')
    taskData = [((f1,), a1) for f1, a1 in taskData[['f1', 'a1']].to_numpy()]
    test.append(makeTask("task1-test", arrow(treal, treal), taskData, 2))

    # Task 2
    taskData = loadTaskDataFromFile('data/aiCompetition/aiCompetitionTasks/task2-train.csv')
    taskData = [((f1, f2), a1) for f1, f2, a1 in taskData[['f1', 'f2', 'a1']].to_numpy()]
    train.append(makeTask("task2-train", arrow(treal, treal, treal), taskData, 2))

    taskData = loadTaskDataFromFile('data/aiCompetition/aiCompetitionTasks/task2-test.csv')
    taskData = [((f1, f2), a1) for f1, f2, a1 in taskData[['f1', 'f2', 'a1']].to_numpy()]
    test.append(makeTask("task2-test", arrow(treal, treal, treal), taskData, 2))

    # Task 3
    taskData = loadTaskDataFromFile('data/aiCompetition/aiCompetitionTasks/task3-train.csv')
    taskData = [((e12, f1, f2), a1) for e12, f1, f2, a1 in taskData[['e12', 'f1', 'f2', 'a1']].to_numpy()]
    train.append(makeTask("task3-train",
                          arrow(treal, treal, treal, treal),
                          taskData,
                          3))

    taskData = loadTaskDataFromFile('data/aiCompetition/aiCompetitionTasks/task3-test.csv')
    taskData = [((e12, f1, f2), a1) for e12, f1, f2, a1 in taskData[['e12', 'f1', 'f2', 'a1']].to_numpy()]
    test.append(makeTask("task3-test",
                         arrow(treal, treal, treal, treal),
                         taskData,
                         3))

    # Task 4
    taskData = loadTaskDataFromFile('data/aiCompetition/aiCompetitionTasks/task4-train.csv')
    taskData = [((ρ, f1, f2), a1) for ρ, f1, f2, a1 in taskData[['ρ', 'f1', 'f2', 'a1']].to_numpy()]
    train.append(makeTask("task4-train",
                          arrow(treal, treal, treal, treal),
                          taskData,
                          3))

    taskData = loadTaskDataFromFile('data/aiCompetition/aiCompetitionTasks/task4-test.csv')
    taskData = [((ρ, f1, f2), a1) for ρ, f1, f2, a1 in taskData[['ρ', 'f1', 'f2', 'a1']].to_numpy()]
    test.append(makeTask("task4-test",
                         arrow(treal, treal, treal, treal),
                         taskData,
                         3))

    # Task 5
#   taskData = loadTaskDataFromFile('data/aiCompetition/aiCompetitionTasks/task5-train.csv')
#   taskData =  [((q, f1, f2), a1) for q, f1, f2, a1 in taskData[['q', 'f1', 'f2', 'a1']].to_numpy()]
#   train.append(makeTask("task5-train",
#                         arrow(treal, treal, treal, treal),
#                         taskData,
#                         3))

#   taskData = loadTaskDataFromFile('data/aiCompetition/aiCompetitionTasks/task5-test.csv')
#   taskData = [((q, f1, f2), a1) for q, f1, f2, a1 in taskData[['q', 'f1', 'f2', 'a1']].to_numpy()]
#   test.append(makeTask("task5-test",
#                        arrow(treal, treal, treal, treal),
#                        taskData,
#                        3))

    # Task 6
    taskData = loadTaskDataFromFile('data/aiCompetition/aiCompetitionTasks/task6-train.csv')
    taskData = [((ρ, e12, f1, f2), a1) for ρ, e12, f1, f2, a1 in taskData[['ρ', 'e12', 'f1', 'f2', 'a1']].to_numpy()]
    train.append(makeTask("task6-train",
                          arrow(treal, treal, treal, treal, treal),
                          taskData,
                          4))

    taskData = loadTaskDataFromFile('data/aiCompetition/aiCompetitionTasks/task6-test.csv')
    taskData = [((ρ, e12, f1, f2), a1) for ρ, e12, f1, f2, a1 in taskData[['ρ', 'e12', 'f1', 'f2', 'a1']].to_numpy()]
    test.append(makeTask("task6-test",
                         arrow(treal, treal, treal, treal, treal),
                         taskData,
                         4))

#   # Task 7
#   taskData = loadTaskDataFromFile('data/aiCompetition/aiCompetitionTasks/task7-train.csv')
#   taskData = [((q, e12, f1, f2), a1) for q, e12, f1, f2, a1 in taskData[['q', 'e12', 'f1', 'f2', 'a1']].to_numpy()]
#   train.append(makeTask("task7-train",
#                         arrow(treal, treal, treal, treal, treal),
#                         taskData,
#                         4))

#   taskData = loadTaskDataFromFile('data/aiCompetition/aiCompetitionTasks/task7-test.csv')
#   taskData = [((q, e12, f1, f2), a1) for q, e12, f1, f2, a1 in taskData[['q', 'e12', 'f1', 'f2', 'a1']].to_numpy()]
#   test.append(makeTask("task7-test",
#                        arrow(treal, treal, treal, treal, treal),
#                        taskData,
#                        4))

#   # Task 8
#   taskData = loadTaskDataFromFile('data/aiCompetition/aiCompetitionTasks/task8-train.csv')
#   taskData = [((q, ρ, f1, f2), a1) for q, ρ, f1, f2, a1 in taskData[['q', 'ρ', 'f1', 'f2', 'a1']].to_numpy()]
#   train.append(makeTask("task8-train",
#                         arrow(treal, treal, treal, treal, treal),
#                         taskData,
#                         4))

#   taskData = loadTaskDataFromFile('data/aiCompetition/aiCompetitionTasks/task8-test.csv')
#   taskData = [((q, ρ, f1, f2), a1) for q, ρ, f1, f2, a1 in taskData[['q', 'ρ', 'f1', 'f2', 'a1']].to_numpy()]
#   test.append(makeTask("task8-test",
#                        arrow(treal, treal, treal, treal, treal),
#                        taskData,
#                        4))

#   # Task 9
#   taskData = loadTaskDataFromFile('data/aiCompetition/aiCompetitionTasks/task9-train.csv')
#   taskData = [((q, ρ, e12, f1, f2), a1) for q, ρ, e12, f1, f2, a1 in taskData[['q', 'ρ', 'e12', 'f1', 'f2', 'a1']].to_numpy()]
#   train.append(makeTask("task9-train",
#                         arrow(treal, treal, treal, treal, treal, treal),
#                         taskData,
#                         5))

#   taskData = loadTaskDataFromFile('data/aiCompetition/aiCompetitionTasks/task9-test.csv')
#   taskData = [((q, ρ, e12, f1, f2), a1) for q, ρ, e12, f1, f2, a1 in taskData[['q', 'ρ', 'e12', 'f1', 'f2', 'a1']].to_numpy()]
#   test.append(makeTask("task9-test",
#                        arrow(treal, treal, treal, treal, treal, treal),
#                        taskData,
#                        5))

    # Task 10
    taskData = loadTaskDataFromFile('data/aiCompetition/aiCompetitionTasks/task10-train.csv')
    taskData = [((ρ, e12, f1, f2), a1) for ρ, e12, f1, f2, a1 in taskData[['ρ', 'e12', 'f1', 'f2', 'a1']].to_numpy()]
    train.append(makeTask("task10-train",
                          arrow(treal, treal, treal, treal, treal),
                          taskData,
                          4))

    taskData = loadTaskDataFromFile('data/aiCompetition/aiCompetitionTasks/task10-test.csv')
    taskData = [((ρ, e12, f1, f2), a1) for ρ, e12, f1, f2, a1 in taskData[['ρ', 'e12', 'f1', 'f2', 'a1']].to_numpy()]
    test.append(makeTask("task10-test",
                         arrow(treal, treal, treal, treal, treal),
                         taskData,
                         4))

    equationPrimitives = [
        # Possibly add individual ints
        #real,
        f0,
        f1,
        f2,
        real_power,
        real_subtraction,
        real_addition,
        real_division,
        real_multiplication
        ] # + groundTruthSolutions()
         #+ [
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
    outputDirectory = "experimentOutputs/aiCompetition/%s"%timestamp
    os.system("mkdir -p %s"%outputDirectory)

    explorationCompression(baseGrammar,
                           train,
                           outputPrefix="%s/aiCompetition"%outputDirectory,
                           evaluationTimeout=0.1,
                           testingTasks=test,
                           **commandlineArguments(
                               enumerationTimeout=3600,
                               iterations=10,
                               compressor="pypy",
                               featureExtractor=DummyFeatureExtractor,
                               CPUs=numberOfCPUs(),
                               structurePenalty=1.,
                               #recognitionTimeout=7200,
                               helmholtzRatio=0.,
                               a=3,
                               maximumFrontier=5,
                               topK=2,
                               pseudoCounts=30.0))
