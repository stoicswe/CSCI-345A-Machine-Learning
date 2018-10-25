import numpy as np
import numpy.linalg as lg
import numpy.random as random
import matplotlib.pyplot as plt

class QuantumRegister(object):
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.n_states = 2 ** n_qubits
        self.qubits = np.zeros((self.n_states), dtype=complex)
        self.qubits[0] = 1.0
        
    def isset(self, state, n):
        return state & 1<<(self.n_qubits-1-n) != 0
    
    def flip(self, state, n):
        return state ^ 1<<(self.n_qubits-1-n)
    
    def set_qubit(self, n, a, b): # a|0>+b|1>
        tmp_qubits = np.zeros((self.n_states), dtype=complex)
        for state in range(self.n_states):
            current_amplitude = self.qubits[state] + self.qubits[self.flip(state,n)]
            if self.isset(state, n):
                tmp_qubits[state] = current_amplitude*b
            else:
                tmp_qubits[state] = current_amplitude*a
        self.qubits = tmp_qubits
    
    def measure(self):
        probabilities = np.absolute(self.qubits)**2
        return random.choice(len(probabilities), p=probabilities.flatten())
        
    def hadamar(self):
        hadamar = np.zeros((self.n_states, self.n_states))
        for target in range(self.n_states):
            for state in range(self.n_states):
                hadamar[target, state] = (2.**(-self.n_qubits/2.))*(-1)**bin(state & target).count("1")
        self.qubits = hadamar.dot(self.qubits)
        return self
    
    def __str__(self):
        string = ""
        for state in range(self.n_states):
            string += "{0:0>3b}".format(state) + " => {:.2f}".format(self.qubits[state]) + "\n"
        return string[:-1]
    
    def plot(self):
        plt.bar(range(self.n_states), np.absolute(self.qubits), color='k')
        plt.title('Register')
        plt.axis([0,self.n_states,0.0,1.0])
        plt.show()

class QRam(object):
    def __init__(self, accessQubits, dataQubits): 
        self.accessQubits = accessQubits
        self.dataQubits = dataQubits

        self.accessStates = 2 ** accessQubits
        self.dataStates = 2 ** dataQubits
        
        self.memory = np.zeros((self.accessStates, self.dataStates), dtype=complex)
        
    def retrieve(self, access): 
        out = np.zeros((self.dataStates,), dtype=complex)
        for i in range(access.shape[0]):
            out += access[i] ** 2 * self.memory[i,:]
        return out
    
    def store(self, access, data):
        assert access < self.accessStates
        self.memory[access,:] = data

class SwapRegister(QuantumRegister):
    def cswap(self, c, a, b):
        cswap = np.zeros((self.n_states, self.n_states))
        for state in range(self.n_states):
            if self.isset(state, c): 
                if self.isset(state, a) != self.isset(state,b):
                    flipstate = self.flip(self.flip(state, b), a)
                    cswap[state, flipstate] = 1.0
                else:
                    cswap[state, state] = 1.0
            else:
                cswap[state, state] = 1.0
        self.qubits = cswap.dot(self.qubits)
        return self

class DotProductRegister(SwapRegister):
    def dot(self):
        pass

def distance(v1, v2):
    return 1-dot(v1,v2)

def dot(v1, v2):
    register = SwapRegister(3)
    #register.qubits[0:4] = np.array([1./np.sqrt(6), 1./np.sqrt(6), 1./np.sqrt(3), 1./np.sqrt(3)])
    # encode the two points
    # x1
    register.set_qubit(1, v1[0], v1[1])
    # x2
    register.set_qubit(2, v2[0], v2[1])
    #print("Initial")
    #print(register)
    H = 1./np.sqrt(2) * np.array([[1., 1.],[1., -1.]])
    eye = np.eye(2)
    H_t = np.kron(H,np.kron(eye,eye))

    register.qubits = H_t.dot(register.qubits)
    #print("Hadamar")
    #print(register)
    register.cswap(0,1,2)
    #print("c-swap")
    #print(register)
    register.qubits = H_t.dot(register.qubits)
    #print("Hadamar")
    #print(register)

    def avg(values):
        return sum(values)/float(len(values))
    average = avg([register.measure() < 4 for i in range (10000)])
    #print average
    #print("Measure average")
    #print(np.sqrt(average*2-1))
    dot_result = np.sqrt(average*2-1)
    return dot_result

#x = [0.09983341664682815, 0.19866933079506122, 0.29552020666133955, 0.3894183423086505, 0.479425538604203, 0.9974949866040544, 0.9995736030415051, 0.9916648104524686, 0.9738476308781951, 0.9463000876874145]
#y = [0.9950041652780258, 0.9800665778412416, 0.955336489125606, 0.9210609940028851, 0.8775825618903728, 0.0707372016677029, -0.029199522301288815, -0.12884449429552464, -0.2272020946930871, -0.32328956686350335]
#print("Distance:")
#print(distance([x[0], y[0]],[x[1], y[1]]))