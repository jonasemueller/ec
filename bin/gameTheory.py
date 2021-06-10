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

tposint = baseType("posint")
tposreal = baseType("posreal")
tzerotoone = baseType("zerotoone")

def _eq(x): return lambda y: x == y

def _gt(x): return lambda y: x > y

def makeTrainingData(request, law,
                     # Number of examples
                     N=10,
                     # Vector dimensionality
                     D=2,
                     # Maximum absolute value of a random number
                     S=20.):
    from random import random, randint

    def sampleArgument(a, listLength):
        if a.name == "real":
            return random() * S * 2 - S
        elif a.name == "posreal":
            return random() * S
        elif a.name == "posint":
            return float(randint(1, S))
        elif a.name == "zerotoone":
            return random()
        elif a.name == "vector":
            return [random() * S * 2 - S for _ in range(D)]
        elif a.name == "list":
            return [sampleArgument(a.arguments[0], listLength)
                    for _ in range(listLength)]
        else:
            assert False, "unknown argument tp %s" % a

    arguments = request.functionArguments()
    e = []
    for _ in range(N):
        # Length of any requested lists
        l = randint(1, 4)

        xs = tuple(sampleArgument(a, l) for a in arguments)
        y = law(*xs)
        e.append((xs, y))

    return e

def genericType(t):
    if t.name == "real":
        return treal
    elif t.name == "posreal":
        return treal
    elif t.name == "int":
        return tint
    elif t.name == "posint":
        return treal # Because of no int support in solver
    elif t.name == "zerotoone":
        return treal
    elif t.name == "bool":
        return tbool
    elif t.name == "vector":
        return tlist(treal)
    elif t.name == "list":
        return tlist(genericType(t.arguments[0]))
    elif t.isArrow():
        return arrow(genericType(t.arguments[0]),
                     genericType(t.arguments[1]))
    else:
        assert False, "could not make type generic: %s" % t

def split(collection, fraction):
    half = math.floor(len(collection) * fraction)
    return collection[half:], collection[: half]

def makeTasksFromFile(name, request, filename, seed):
    data = pd.read_csv(filename)
    data['a1'] = data['a1'].clip(lower=0.0, upper=1.0)
    data['a2'] = data['a2'].clip(lower=0.0, upper=1.0)
    groupedData = data.groupby(['e12'])
    testTasks = []
    trainTasks = []
    for group, value in groupedData:
        e = [((f1, f2), a1) for f1, f2, a1 in value[['f1', 'f2', 'a1']].to_numpy()]
        random.Random(seed).shuffle(e)
        test, train = split(tuple(e), 0.5)
        testTasks.append(DifferentiableTask(name + " enmity " + str(group),
                                            genericType(request),
                                            test,
                                            BIC=1.,
                                            restarts=2,
                                            steps=25,
                                            maxParameters=2,
                                            loss=squaredErrorLoss))
        trainTasks.append(DifferentiableTask(name + " enmity " + str(group),
                                             genericType(request),
                                             train,
                                             BIC=1.,
                                             restarts=2,
                                             steps=25,
                                             maxParameters=2,
                                             loss=squaredErrorLoss))
    return testTasks, trainTasks

def makeTasksFromFile2(name, request, filename, seed):
    data = pd.read_csv(filename)
    data['a1'] = data['a1'].clip(lower=0.0, upper=1.0)
    data['a2'] = data['a2'].clip(lower=0.0, upper=1.0)
    testTasks = []
    trainTasks = []
    e = [((e12, f1, f2), a1) for e12, f1, f2, a1 in data[['e12', 'f1', 'f2', 'a1']].to_numpy()]
    random.Random(seed).shuffle(e)
    test, train = split(tuple(e), 0.1)
    testTasks.append(DifferentiableTask(name + " w. enmity as parameter ",
                                        genericType(request),
                                        test,
                                        BIC=1.,
                                        restarts=2,
                                        steps=25,
                                        maxParameters=3,
                                        loss=squaredErrorLoss))
    trainTasks.append(DifferentiableTask(name + " w. enmity as parameter ",
                                         genericType(request),
                                         train,
                                         BIC=1.,
                                         restarts=2,
                                         steps=25,
                                         maxParameters=3,
                                         loss=squaredErrorLoss))
    return testTasks, trainTasks

def makeTasksFromFile3(name, request, data_transform, filename):
    data = pd.read_csv(filename)
    data = data[data.status.eq('Solve_Succeeded')]
    #print(len(data[data.status.ne('Solve_Succeeded')]))
    data['a2'] = data['a2'].clip(lower=0.0, upper=1.0)
    data['a1'] = data['a1'].clip(lower=0.0, upper=1.0)
    return DifferentiableTask(name,
                              genericType(request),
                              data_transform(data),
                              BIC=1.,
                              restarts=2,
                              steps=25,
                              maxParameters=5,
                              loss=squaredErrorLoss)

def makeTask(name, request, law,
             # Number of examples
             N=50,
             # Vector dimensionality
             D=3,
             # Maximum absolute value of a random number
             S=10.):
    e = makeTrainingData(request, law,
                         N=N, D=D, S=S)

    return DifferentiableTask(name, genericType(request), e,
                              BIC=10.,
                              likelihoodThreshold=-0.001,
                              restarts=2,
                              steps=25,
                              maxParameters=1,
                              loss=squaredErrorLoss)

#class LearnedFeatureExtractor(RecurrentFeatureExtractor):
#   def tokenize(self, examples):
#       # Should convert both the inputs and the outputs to lists
#       def t(z):
#           if isinstance(z, list):
#               return ["STARTLIST"] + \
#                   [y for x in z for y in t(x)] + ["ENDLIST"]
#           assert isinstance(z, (float, int))
#           return ["REAL"]
#       return [(tuple(map(t, xs)), t(y))
#               for xs, y in examples]

#   def __init__(self, tasks, examples, testingTasks=[], cuda=False):
#       lexicon = {c
#                  for t in tasks + testingTasks
#                  for xs, y in self.tokenize(t.examples)
#                  for c in reduce(lambda u, v: u + v, list(xs) + [y])}

#       super(LearnedFeatureExtractor, self).__init__(lexicon=list(lexicon),
#                                                     cuda=cuda,
#                                                     H=64,
#                                                     tasks=tasks,
#                                                     bidirectional=True)

#   def featuresOfProgram(self, p, tp):
#       p = program.visit(RandomParameterization.single)
#       return super(LearnedFeatureExtractor, self).featuresOfProgram(p, tp)

def noInformationUnboundedPolicy(mu, e, n):
    return mu / (e * n)

def noInformationPolicy(mu, e, n):
  if mu < e * n:
      return noInformationUnboundedPolicy(mu, e, n)
  else:
      return 1.

def privateInformationUnboundedPolicy(x, mu, e, n):
    return x / (e * n - e + 1.)

def privateInformationPolicy(x, mu, e, n):
    if x < e * n - e + 1.:
        return privateInformationUnboundedPolicy(x, mu, e, n)
    else:
        return 1.

def fullInformationUnboundedPolicy(delta, mu, e, n):
    return delta / e

def fullInformationPolicy(delta, mu, e, n):
    if delta / e < 1.:
        return fullInformationUnboundedPolicy(delta, mu, e, n)
    else:
        return 1.

if __name__ == "__main__":

    test = []
    train = []

    # Task 1
    test.append(makeTasksFromFile3("PrRTTP task 1 testing",
                                   arrow(tposreal, tposreal, tposreal, tposreal),
                                   lambda data : [((e12, f1, f2), a1) for e12, f1, f2, a1 in data[['e12', 'f1', 'f2', 'a1']].to_numpy()],
                                   'prrttp/simulations/task1-test.csv'))
    train.append(makeTasksFromFile3("PrRTTP task 1 train",
                                    arrow(tposreal, tposreal, tposreal, tposreal),
                                    lambda data : [((e12, f1, f2), a1) for e12, f1, f2, a1 in data[['e12', 'f1', 'f2', 'a1']].to_numpy()],
                                    'prrttp/simulations/task1-train.csv'))

    # Task 2
    test.append(makeTasksFromFile3("PrRTTP task 2 testing",
                                   arrow(tposreal, tposreal, tposreal, tposreal),
                                   lambda data : [((ρ, f1, f2), a1) for ρ, f1, f2, a1 in data[['ρ', 'f1', 'f2', 'a1']].to_numpy()],
                                   'prrttp/simulations/task2-test.csv'))
    train.append(makeTasksFromFile3("PrRTTP task 2 train",
                                    arrow(tposreal, tposreal, tposreal, tposreal),
                                    lambda data : [((ρ, f1, f2), a1) for ρ, f1, f2, a1 in data[['ρ', 'f1', 'f2', 'a1']].to_numpy()],
                                    'prrttp/simulations/task2-train.csv'))

    # Task 3
    test.append(makeTasksFromFile3("PrRTTP task 3 testing",
                                   arrow(tposreal, tposreal, tposreal, tposreal),
                                   lambda data : [((q, f1, f2), a1) for q, f1, f2, a1 in data[['q', 'f1', 'f2', 'a1']].to_numpy()],
                                   'prrttp/simulations/task3-test.csv'))
    train.append(makeTasksFromFile3("PrRTTP task 3 train",
                                    arrow(tposreal, tposreal, tposreal, tposreal),
                                    lambda data : [((q, f1, f2), a1) for q, f1, f2, a1 in data[['q', 'f1', 'f2', 'a1']].to_numpy()],
                                    'prrttp/simulations/task3-train.csv'))

    # Task 4
    test.append(makeTasksFromFile3("PrRTTP task 4 testing",
                                   arrow(tposreal, tposreal, tposreal, tposreal, tposreal),
                                   lambda data : [((ρ, e12, f1, f2), a1) for ρ, e12, f1, f2, a1 in data[['ρ', 'e12', 'f1', 'f2', 'a1']].to_numpy()],
                                   'prrttp/simulations/task4-test.csv'))
    train.append(makeTasksFromFile3("PrRTTP task 4 train",
                                    arrow(tposreal, tposreal, tposreal, tposreal, tposreal),
                                    lambda data : [((ρ, e12, f1, f2), a1) for ρ, e12, f1, f2, a1 in data[['ρ', 'e12', 'f1', 'f2', 'a1']].to_numpy()],
                                    'prrttp/simulations/task4-train.csv'))

    # Task 5
    test.append(makeTasksFromFile3("PrRTTP task 5 testing",
                                   arrow(tposreal, tposreal, tposreal, tposreal, tposreal),
                                   lambda data : [((q, e12, f1, f2), a1) for q, e12, f1, f2, a1 in data[['q', 'e12', 'f1', 'f2', 'a1']].to_numpy()],
                                   'prrttp/simulations/task5-test.csv'))
    train.append(makeTasksFromFile3("PrRTTP task 5 train",
                                    arrow(tposreal, tposreal, tposreal, tposreal, tposreal),
                                    lambda data : [((q, e12, f1, f2), a1) for q, e12, f1, f2, a1 in data[['q', 'e12', 'f1', 'f2', 'a1']].to_numpy()],
                                    'prrttp/simulations/task5-train.csv'))

    # Task 6
    test.append(makeTasksFromFile3("PrRTTP task 6 testing",
                                   arrow(tposreal, tposreal, tposreal, tposreal, tposreal),
                                   lambda data : [((q, ρ, f1, f2), a1) for q, ρ, f1, f2, a1 in data[['q', 'ρ', 'f1', 'f2', 'a1']].to_numpy()],
                                   'prrttp/simulations/task6-test.csv'))
    train.append(makeTasksFromFile3("PrRTTP task 6 train",
                                    arrow(tposreal, tposreal, tposreal, tposreal, tposreal),
                                    lambda data : [((q, ρ, f1, f2), a1) for q, ρ, f1, f2, a1 in data[['q', 'ρ', 'f1', 'f2', 'a1']].to_numpy()],
                                    'prrttp/simulations/task6-train.csv'))

    # Task 7
    test.append(makeTasksFromFile3("PrRTTP task 7 testing",
                                   arrow(tposreal, tposreal, tposreal, tposreal, tposreal, tposreal),
                                   lambda data : [((q, ρ, e12, f1, f2), a1) for q, ρ, e12, f1, f2, a1 in data[['q', 'ρ', 'e12', 'f1', 'f2', 'a1']].to_numpy()],
                                   'prrttp/simulations/task7-test.csv'))
    train.append(makeTasksFromFile3("PrRTTP task 7 train",
                                    arrow(tposreal, tposreal, tposreal, tposreal, tposreal, tposreal),
                                    lambda data : [((q, ρ, e12, f1, f2), a1) for q, ρ, e12, f1, f2, a1 in data[['q', 'ρ', 'e12', 'f1', 'f2', 'a1']].to_numpy()],
                                    'prrttp/simulations/task7-train.csv'))

#    seed = 1234
#    test1, train1 = makeTasksFromFile("PrRTTP",
#                                      arrow(tposreal, tposreal, tposreal),
#                                      "prrttp/results_rho20_fmany_df.csv",
#                                      seed)
#    test2, train2 = makeTasksFromFile2("PrRTTP",
#                                       arrow(tzerotoone, tposreal, tposreal, tposreal),
#                                       "prrttp/results_rho20_fmany_df.csv",
#                                       seed)
#    test = test1 + test2
#    train = train1 + train2
#    eprint("Training on", len(train), "tasks")
#    eprint("Testing on", len(test), "tasks")

##    tasks = [
# Note: Can't send booleans to solver

# Solutions
#        makeTask("1st choice no info",
#                 arrow(tposreal, tzerotoone, tposint, tposreal),
#                 noInfoUnboundedPolicy),
#        makeTask("choice point no info concept",
#                 arrow(tposreal, tzerotoone, tposint, tbool),
#                 lambda mu, e, n: mu < e * n),
#        makeTask("choice concept no info",
#                 arrow(tposreal, tzerotoone, tposint, tposreal),
#                 lambda mu, e, n: e * n),
#       makeTask("choice point no info",
#                arrow(tposreal, tzerotoone, tposint, tposreal),
#                lambda mu, e, n: mu < e * n),
##        makeTask("No information policy",
##                 arrow(tposreal, tzerotoone, tposint, tposreal),
##                 noInformationPolicy),
#        makeTask("1st choice private info",
#                 arrow(tposreal, tposreal, tzerotoone, tposint, tposreal),
#                 lambda x, mu, e, n: x / (e * n - e + 1.)),
#        makeTask("choice concept private info",
#                 arrow(tposreal, tposreal, tzerotoone, tposint, tposreal),
#                 lambda x, mu, e, n: e * n - e + 1.),
#       makeTask("choice point private info",
#                arrow(tposreal, tposreal, tzerotoone, tposint, tposreal),
#                lambda x, mu, e, n: x < e * n - e + 1.),
##        makeTask("Private information policy",
##                 arrow(tposreal, tposreal, tzerotoone, tposint, tposreal),
##                 privateInformationPolicy),

#        makeTask("1st choice public info",
#                 arrow(tposreal, tposreal, tzerotoone, tposint, tposreal),
#                 lambda delta, mu, e, n: delta/e),
#        makeTask("choice point public info",
#                 arrow(tposreal, tposreal, tzerotoone, tposint, tposreal),
#                 lambda delta, mu, e, n: delta/e < 1.),
##        makeTask("Public information policy",
##                 arrow(tposreal, tposreal, tzerotoone, tposint, tposreal),
##                 fullInformationPolicy),

        # Disaster probabilities
#        makeTask("1st choice public disaster",
#                 arrow(tposreal, tzerotoone, tposint, tposreal),
#                 lambda mu, e, n: 1 - mu / (e * (n + 1))),
#        makeTask("2nd choice public disaster",
#                 arrow(tposreal, tzerotoone, tposint, tposreal),
#                 lambda mu, e, n: ((mu - e)**(n+1)) / (e*(n+1)*mu**n) - (mu / (e * (n + 1))) + 1),
#        makeTask("public disaster",
#                 arrow(tposreal, tzerotoone, tposint, tposreal),
#                 lambda mu, e, n: 1 - mu / (e * (n + 1)) if mu < e else ((mu - e)**(n+1)) / (e*(n+1)*mu**n) - (mu / (e * (n + 1))) + 1),
 ##   ]
    #bootstrapTarget()
    #real_gt = Primitive("gt?.", arrow(treal, treal, tbool), _gt)
    #real_eq =  Primitive("eq?.", arrow(treal, treal, tbool), _eq)
    equationPrimitives = [
        # Possibly add individual ints
        #real,
        f0,
        f1,
        real_subtraction,
        real_addition,
        real_division,
        real_multiplication,
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
#    baseGrammar = Grammar.uniform(equationPrimitives + [p for p in bootstrapTarget()])

#    eprint("Got %d equation discovery tasks..." % len(tasks))

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
                               structurePenalty=0.5,
                               helmholtzRatio=0.5,
                               a=3,
                               maximumFrontier=10000,
                               topK=2,
                               pseudoCounts=10.0))
