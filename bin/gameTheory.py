import datetime
import os
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
from dreamcoder.program import Program
from dreamcoder.recognition import RecurrentFeatureExtractor, DummyFeatureExtractor
from dreamcoder.task import DifferentiableTask, squaredErrorLoss
from dreamcoder.type import baseType, tlist, arrow
from dreamcoder.utilities import eprint, numberOfCPUs

tvector = baseType("vector")
treal = baseType("real")
tpositive = baseType("positive")


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
        elif a.name == "positive":
            return random() * S
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


def makeTask(name, request, law,
             # Number of examples
             N=20,
             # Vector dimensionality
             D=3,
             # Maximum absolute value of a random number
             S=20.):
    print(name)
    e = makeTrainingData(request, law,
                         N=N, D=D, S=S)
    print(e)
    print()

    def genericType(t):
        if t.name == "real":
            return treal
        elif t.name == "positive":
            return treal
        elif t.name == "vector":
            return tlist(treal)
        elif t.name == "list":
            return tlist(genericType(t.arguments[0]))
        elif t.isArrow():
            return arrow(genericType(t.arguments[0]),
                         genericType(t.arguments[1]))
        else:
            assert False, "could not make type generic: %s" % t

    return DifferentiableTask(name, genericType(request), e,
                              BIC=10.,
                              likelihoodThreshold=-0.001,
                              restarts=2,
                              steps=25,
                              maxParameters=1,
                              loss=squaredErrorLoss)


def norm(v):
    return sum(x * x for x in v)**0.5


def unit(v):
    return scaleVector(1. / norm(v), v)


def scaleVector(a, v):
    return [a * x for x in v]


def innerProduct(a, b):
    return sum(x * y for x, y in zip(a, b))


def crossProduct(a, b):
    (a1, a2, a3) = tuple(a)
    (b1, b2, b3) = tuple(b)
    return [a2 * b3 - a3 * b2,
            a3 * b1 - a1 * b3,
            a1 * b2 - a2 * b1]


def vectorAddition(u, v):
    return [a + b for a, b in zip(u, v)]

def vectorSubtraction(u, v):
    return [a - b for a, b in zip(u, v) ]


class LearnedFeatureExtractor(RecurrentFeatureExtractor):
    def tokenize(self, examples):
        # Should convert both the inputs and the outputs to lists
        def t(z):
            if isinstance(z, list):
                return ["STARTLIST"] + \
                    [y for x in z for y in t(x)] + ["ENDLIST"]
            assert isinstance(z, (float, int))
            return ["REAL"]
        return [(tuple(map(t, xs)), t(y))
                for xs, y in examples]

    def __init__(self, tasks, examples, testingTasks=[], cuda=False):
        lexicon = {c
                   for t in tasks + testingTasks
                   for xs, y in self.tokenize(t.examples)
                   for c in reduce(lambda u, v: u + v, list(xs) + [y])}

        super(LearnedFeatureExtractor, self).__init__(lexicon=list(lexicon),
                                                      cuda=cuda,
                                                      H=64,
                                                      tasks=tasks,
                                                      bidirectional=True)

    def featuresOfProgram(self, p, tp):
        p = program.visit(RandomParameterization.single)
        return super(LearnedFeatureExtractor, self).featuresOfProgram(p, tp)


if __name__ == "__main__":
    pi = 3.14  # I think this is close enough to pi
    # Data taken from:
    # https://secure-media.collegeboard.org/digitalServices/pdf/ap/ap-physics-1-equations-table.pdf
    # https://secure-media.collegeboard.org/digitalServices/pdf/ap/physics-c-tables-and-equations-list.pdf
    # http://mcat.prep101.com/wp-content/uploads/ES_MCATPhysics.pdf
    # some linear algebra taken from "parallel distributed processing"
    tasks = [
# Solutions
        makeTask("1st choice no info",
                 arrow(tpositive, tpositive, tpositive, tpositive),
                 lambda mu, e, n: mu / (e*n)),
#        makeTask("choice point no info concept",
#                 arrow(tpositive, tpositive, tpositive, tpositive),
#                 lambda mu, e, n: e * n),
#        makeTask("choice point no info",
#                 arrow(tpositive, tpositive, tpositive, tpositive),
#                 lambda mu, e, n: 0 if mu < e * n else 1),
#        makeTask("full no info",
#                 arrow(tpositive, tpositive, tpositive, tpositive),
#                 lambda mu, e, n:  mu / (e*n) if mu < e * n else 1),

#        makeTask("2nd choice no info",
#                 arrow(tpositive, tpositive, tpositive, tpositive),
#                 lambda mu, e, n: 1),
        makeTask("1st choice private info",
                 arrow(tpositive, tpositive, tpositive, tpositive, tpositive),
                 lambda x, mu, e, n: x / (e * n - e + 1)),
#        makeTask("choice point private info",
#                 arrow(tpositive, tpositive, tpositive, tpositive, tpositive),
#                 lambda x, mu, e, n: e * n - e + 1),
#        makeTask("2nd choice private info",
#                arrow(tpositive, tpositive, tpositive, tpositive, tpositive),
#                lambda x, mu, e, n: 1),
        makeTask("1st choice public info",
                 arrow(tpositive, tpositive, tpositive, tpositive, tpositive),
                 lambda delta, mu, e, n: delta/e),
#        makeTask("choice point public info",
#                 arrow(tpositive, tpositive, tpositive, tpositive, tpositive),
#                 lambda delta, mu, e, n: delta/e),
#      makeTask("2nd choice public info",
#               arrow(tpositive, tpositive, tpositive, tpositive, tpositive),
#               lambda delta, mu, e, n: 1),
# Disaster probabilities
       makeTask("1st choice public disaster",
                arrow(tpositive, tpositive, tpositive, treal),
                lambda mu, e, n: 1 - mu / (e * (n + 1))),
       makeTask("2nd choice public disaster", # TODO introduce rand int, can't do power of fraction
                arrow(tpositive, tpositive, tpositive, treal),
                lambda mu, e, n: ((mu - e)**(round(n)+1)) / (e*(n+1)*mu**round(n)) - (mu / (e * (n + 1))) + 1),
       makeTask("public disaster",
                arrow(tpositive, tpositive, tpositive, tpositive),
                lambda mu, e, n: 1 - mu / (e * (n + 1)) if mu < e else ((mu - e)**(round(n)+1)) / (e*(n+1)*mu**round(n)) - (mu / (e * (n + 1))) + 1),
    ]
    bootstrapTarget()
    equationPrimitives = [
        real,
        f0,
        f1,
        fpi,
        real_power,
        real_subtraction,
        real_addition,
        real_division,
        real_multiplication] + [
            Program.parse(n)
            for n in ["map","fold",
                      "empty","cons","car","cdr",
                      "zip",
                      "unfold", "range", "index", "length", "if", "empty?",]]
    baseGrammar = Grammar.uniform(equationPrimitives)
#    baseGrammar = Grammar.uniform(equationPrimitives + [p for p in bootstrapTarget()])

    eprint("Got %d equation discovery tasks..." % len(tasks))

    timestamp = datetime.datetime.now().isoformat()
    outputDirectory = "experimentOutputs/gameTheory/%s"%timestamp
    os.system("mkdir -p %s"%outputDirectory)

    explorationCompression(baseGrammar, tasks,
                           outputPrefix="%s/gameTheory"%outputDirectory,
                           evaluationTimeout=0.1,
                           testingTasks=[],
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
