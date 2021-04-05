import datetime
import os
import pandas as pd
import random
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

def makeTasksFromFile(name, request, filename):
    data = pd.read_csv(filename)
    data['a1'] = data['a1'].clip(lower=0.0, upper=1.0)
    data['a2'] = data['a2'].clip(lower=0.0, upper=1.0)
    groupedData = data.groupby(['e12'])
    tasks = []
    for group, value in groupedData:
        e = tuple(((f1, f2), a1) for f1, f2, a1 in value[['f1', 'f2', 'a1']].to_numpy())
        tasks.append(DifferentiableTask("PrRTTP enmity " + str(group),
                                        genericType(request),
                                        e,
                                        BIC=10.,
                                        likelihoodThreshold=-0.001,
                                        restarts=2,
                                        steps=25,
                                        maxParameters=1,
                                        loss=squaredErrorLoss))
    return tasks

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
    tasks = makeTasksFromFile("Test",
                              arrow(tposreal, tposreal, tposreal),
                              "prrttp/results_rho20_fmany_df.csv")
    random.shuffle(tasks)
    test, train = testTrainSplit(tasks, 10)
    eprint("Training on", len(train), "tasks")
    eprint("Testing on", len(test), "tasks")

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

    explorationCompression(baseGrammar, train,
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
