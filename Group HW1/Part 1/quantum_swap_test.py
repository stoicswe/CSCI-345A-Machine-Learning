#%matplotlib inline

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
        
register = QuantumRegister(3)
register.hadamar()
print(register.measure())

register = QuantumRegister(2)
register.set_qubit(0, 1/np.sqrt(2), 1/np.sqrt(2))
print(register)
register.set_qubit(1, 1/np.sqrt(3), np.sqrt(2./3.))
print(register)

register.plot()