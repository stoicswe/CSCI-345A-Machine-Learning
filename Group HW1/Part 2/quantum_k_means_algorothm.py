# based on IBM's QISKIT Quantum API

# import math lib
from math import pi
import math

# import Qiskit
from qiskit import Aer, IBMQ, execute
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

# import basic plot tools
from qiskit.tools.visualization import plot_histogram

# To use local qasm simulator
backend = Aer.get_backend('qasm_simulator')
theta_list = [0.1, 0.2, 0.3, 0.4, 0.5, 1.5, 1.6, 1.7, 1.8, 1.9] 

# create Quantum Register called "qr" with 5 qubits
qr = QuantumRegister(5, name="qr")
# create Classical Register called "cr" with 5 bits
cr = ClassicalRegister(5, name="cr")
    
# Creating Quantum Circuit called "qc" involving your Quantum Register "qr"
# and your Classical Register "cr"
qc = QuantumCircuit(qr, cr, name="k_means")
    
result_values = []
#Define a loop to compute the distance between each pair of points
for i in range(9):
    for j in range(1,10-i):
        # Set the parament theta about different point
        theta_1 = theta_list[i]
        theta_2 = theta_list[i+j]
        #Achieve the quantum circuit via qiskit
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
        #print(result)
        #print('theta_1:' + str(theta_1))
        #print('theta_2:' + str(theta_2))
        #print(result.get_counts())
        result_values = [theta_1, theta_2]
        for r in result_values:
                print("{0}, {1}".format(math.sin(r), math.cos(r)))
        #print(result.get_data(qc))
        #plot_histogram(result.get_counts())