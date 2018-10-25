from math import pi
import math
from qiskit import Aer, IBMQ, execute
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.tools.visualization import plot_histogram
from linear_algebra_hybrid import vector_mean
import random

backend = Aer.get_backend('qasm_simulator')
qr = QuantumRegister(5, name="qr")
cr = ClassicalRegister(5, name="cr")
qcircuit = QuantumCircuit(qr, cr, name="k_means")

class KMeans:
    """performs k-means clustering"""

    def __init__(self, k):
        self.k = k          # number of clusters
        self.means = None   # means of clusters

    def classify(self, input):
        """return the index of the cluster closest to the input"""
        #print(math.asin(self.means[0][0]))
        return min(range(self.k),
                   key=lambda i: squared_distance(math.asin(input[0]), math.asin(self.means[i][0])))

    def train(self, thetas):
        inputs = [[math.sin(t), math.cos(t)] for t in thetas]
        #print(inputs)
        self.means = random.sample(inputs, self.k)
        assignments = None

        while True:
            # Find new assignments
            new_assignments = list(map(self.classify, inputs))

            # If no assignments have changed, we're done.
            if assignments == new_assignments:
                return

            # Otherwise keep the new assignments,
            assignments = new_assignments

            for i in range(self.k):
                i_points = [p for p, a in zip(inputs, assignments) if a == i]
                # avoid divide-by-zero if i_points is empty
                #this is where the centroids are and need to
                #be normalized for quantum states..
                if i_points:
                    self.means[i] = vector_mean(i_points)

def squared_distance(theta1, theta2):
    return dist(theta1, theta2)**2
    
def dist(theta_1, theta_2, qc=qcircuit, accuracy=10):
    result_values = []
    for i in range(accuracy):
        qc.h(qr[2])
        qc.h(qr[1])
        qc.h(qr[4])
        qc.u3(theta_1, pi, pi, qr[1])
        qc.u3(theta_2, pi, pi, qr[4])
        qc.cswap(qr[2], qr[1], qr[4])
        qc.h(qr[2])

        qc.measure(qr[2], cr[2])
        qc.reset(qr)
                    
        job = execute(qc, backend=backend, shots=1024)
        result = job.result()
        result_values = [theta_1, theta_2]
        qresult = result.get_counts()
        totalc = sum(list(result.get_counts().values()))
        try:
            distance = qresult['00100'] / totalc
        except:
            distance = 0
        #print(result.get_counts())
        #thetas = [theta_1, theta_2]
        #k = 0
        #for r in result_values:
        #    print("Theta {0}: {1}".format(k, thetas[k]))
        #    print("{0}, {1}".format(math.sin(r), math.cos(r)))
        #    k += 1
        #print("Distance: ", distance)
        result_values.append(distance)
    distance = float("{0:.14f}".format(sum(result_values) / accuracy))
    return distance

theta_list = [0.1, 0.2, 0.3, 0.4, 0.5, 1.5, 1.6, 1.7, 1.8, 1.9]
#print("Distance: ", dist(theta_list[0], theta_list[1]))
kmean = KMeans(2)
kmean.train(theta_list)
#kmean.classify(0.2)