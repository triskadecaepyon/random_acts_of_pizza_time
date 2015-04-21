import numpy as np
from feature_set import FeatureSet
from features.ngram_feature_set import NGramFeatureSet
import copy
import math

featureSet = NGramFeatureSet()
X = featureSet.smartLoad("data_cache/unigrams.npa", binary = False, lowercase = True, min_df=5, ngram_range = (1, 1))
print X

#testMatrix = np.array([[0,0,0],[0,1,0],[0,0,1]])
'''foundDeficient = False
while not foundDeficient:
    testMatrix = np.random.rand(3, 3)
    print testMatrix
    foundDeficient = np.linalg.matrix_rank(testMatrix) != 3
'''

def printColumnVectors(matrix):
    vectorStrings = []
    for vector in matrix.T:
        vectorStrings.append('{%s}' % ','.join(vector.astype(str)))
    print '{%s}' % ','.join(vectorStrings)

def projectOntoSubspace(vector, subspace):
    projection = np.zeros(len(vector))
    for basisVector in subspace.T:
        projection += np.inner(vector, basisVector) * basisVector

    return projection

def distanceToSubspace(vector, subspace):
    return np.linalg.norm(vector - projectOntoSubspace(vector, subspace), 2)

subspace = np.array([[1, 0]]).T
vector = np.array([3, 3])
print projectOntoSubspace(vector, subspace)
print distanceToSubspace(vector, subspace)

def generateRandomSubspace(dimension, enclosingDimension):
    print "generating first"
    basis = np.random.rand(enclosingDimension, dimension)

    print "checking rank"
    # This shouldn't take long to find
    while np.linalg.matrix_rank(basis) != dimension:
        print "trying again"
        basis = np.random.rand(enclosingDimension, dimension)
        
    print "applying gramschmidt"
    for i in range(dimension):
        u = basis[:, i]
        for j in range(i):
            v = basis[:, j]
            u = u - (np.inner(u, v) / np.inner(v, v)) * v
    
        basis[:, i] = u
    
    print "applying normalization"
    for i in range(dimension):
        v = basis[:, i]
        basis[:, i] = v / np.linalg.norm(v)

    print basis.shape
    return basis

printColumnVectors(generateRandomSubspace(3, 3))



# Expects an (n x v) matrix, where n corresponds to the documents and v corresponds to the words
def wordCooccurences(h):
    (n, v) = h.shape
    h_diag = np.zeros(v)
    h_norm = np.zeros((n, v)) 
    for i in range(len(h)):
        h_d = h[i]
        n_d = h_d.sum()
        norm = n_d * (n_d - 1) # @TODO: Play with this. Subtracting 1 seems to be wrong, but that's what the paper says :P
        h_diag_d = h_d / norm 
        h_norm_d = h_d / norm ** 0.5
        h_diag += h_diag_d
        h_norm[i] = h_norm_d

    return np.dot(h_norm.T, h_norm) - np.diag(h_diag)


def computeSubspaceDimension(V, t_a):
    # @TODO: experiment with where to put the division
    # @TODO: what should I set the t_a parameter to?
    return int(4 * math.log(V / t_a ** 2))

def fastAnchorWords(Q, K, t_a):
    V = len(Q)
    subspaceDimension = computeSubspaceDimension(V, t_a)
    subspace = generateRandomSubspace(subspaceDimension, V)

    S = []
    S_0 = None
    maxDistance = -1.0
    for i in range(V): 
        Q[i] = projectOntoSubspace(Q[i], subspace)
        distance = np.linalg.norm(Q[i], 2)
        if distance > maxDistance:
            S_0 = i
            maxDistance = distance

    S.append(S_0)

    for i in range(K - 1):
        S_i = None
        maxDistance = -1.0
        for j in range(V): 

            Q_s = Q[S].T
            d = Q[j]
            distance = distanceToSubspace(d, Q_s)
            # @TODO: I shouldn't have to check if i is not in S
            if j not in S and distance > maxDistance:
                S_i = j
                maxDistance = distance

        S.append(S_i)

    for i in range(K):
        tempS = copy.copy(S)
        tempS.pop(i)
        r_i = None
        maxDistance = -1.0
        for j in range(V): 
            print j
            Q_s = Q[tempS].T
            d = Q[j]
            distance = distanceToSubspace(d, Q_s)
            # @TODO: I shouldn't have to check if i is not in S
            if j not in S and distance > maxDistance:
                r_i = j
                maxDistance = distance
        S[i] = r_i

    return S

def normalize(Q):
    norms = np.zeros(len(Q))
    for i in range(len(Q)):
        norms[i] = np.linalg.norm(Q[i], 1)
        Q[i] /= norms[i] 

    return (Q, norms)

def klGradient(Q_i, Q_s, x):
    K = len(Q_s)
    denom = np.dot(x, Q_s)
    g = np.zeros(K)
    for k in range(K):
        g[k] = (Q_i * Q_s[k] / denom).sum()

    return g

step = 0.0001
def recoverC(Q_i, Q_s, t_b):
    K = len(Q_s)
    x = [np.zeros(K)]
    t = 0
    x[t].fill(1.0 / K)
    converged = False

    prevTest = 0
    while not converged:
        t += 1
        g = klGradient(Q_i, Q_s, x[t - 1])
        temp = np.zeros(K)
        temp.fill(math.e)
        x.append(x[t - 1] * np.power(temp, -step * g))
        x[t] /= np.linalg.norm(x[t], 1)
        g_t = klGradient(Q_i, Q_s, x[t])
        mu_t = -g_t.min()
        lambda_t = g_t + mu_t
        test = np.dot(lambda_t.T, x[t])
        if prevTest != test:
            print test
            prevTest = test
        
        converged =  test < t_b

    return x[t]

def recoverKL(Q, norms, S, t_b):
    V = len(Q)
    K = len(S)
    Q_s = Q[S]
    C = np.zeros((V, K))
    for i in range(V):
        C[i] = recoverC(Q[i], Q_s, t_b)
    A = np.dot(np.diag(norms), C)
    (A, norms) = normalize(A)
    A_pinv = np.linalg.pinv(A)
    R = np.dot(np.dot(A_pinv, Q), A_pinv.T)
    return (A, R)

h = np.array([[1, 1, 2], [1, 0, 2], [1, 4, 0], [1, 1, 1]], dtype=float)
Q = wordCooccurences(h)
(Q, norms) = normalize(Q)
#print recoverKL(Q, norms, [0,1], None)

def high_level_algorithm(D, K, t_a, t_b):
    Q = wordCooccurences(D)
    (Q, norms) = normalize(Q) # @TODO: I normalized, but still not sure if it's right!
    S = fastAnchorWords(Q, K, t_a)
    A, R = recoverKL(Q, norms, S, t_b)
    return A, R

print high_level_algorithm(X, 20, 0.01, 4.0)
    
