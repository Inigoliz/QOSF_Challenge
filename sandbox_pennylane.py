import numpy as np
import scipy.stats
import pennylane as qml

random_U = scipy.stats.unitary_group.rvs(3)
print(random_U)

n_wires = 2
pauli_x = [[0, 1], [1, 0]]
pauli_z = [[1, 0], [0, -1]]
pauli_z_2 = np.kron(pauli_z, pauli_z)


def U():
    # this may not generate a uniform distribution of states, but certainly can yield any state
    #random_U = scipy.stats.unitary_group.rvs(3)
    U = np.kron(pauli_x, pauli_x)
    qml.QubitUnitary(U, wires=[0, 1])


dev = qml.device("default.qubit", wires=n_wires, analytic=True, shots=1)


@qml.qnode(dev)
def circuit(edge=None):
    U()
    return qml.expval(qml.Hermitian(pauli_z_2, wires=[0, 1]))


print(circuit())
