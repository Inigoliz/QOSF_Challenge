import pennylane as qml
from pennylane import numpy as np

n_wires = 4  # number of qubits


def norm(x):
    """ Norm of a quantum state: ||x|| = (<x|x>)^1/2 """
    return np.sqrt(np.sum(np.absolute(x)**2))


def Even(thetaEven):
    """ Applies a set of RZ(theta) to all the qubits """
    for wire in range(n_wires):
        qml.RX(thetaEven[wire], wires=wire)


def Odd(thetaOdd):
    """ Applies a set of RZ(theta) followed by CZ entanglement between every pair of qubits
    """
    for wire in range(n_wires):
        qml.RZ(thetaOdd[wire], wires=wire)
        for i in range(n_wires):
            for j in range(i+1, n_wires):
                qml.CZ(wires=[i, j])


def density_matrix(state):
    """  Calculates the density matrix representation of a state (a state is a complex vector
         in the canonical basis representation). The density matrix is a Hermitian operator.
    """
    return np.outer(np.conj(state), state)


dev = qml.device("default.qubit", wires=n_wires)


@ qml.qnode(dev)
def circuit(thetaEven_s, thetaOdd_s, n_layers=1):
    """ Combines the different layers and outputs F^2 (Fidelity squared)
    """
    for i in range(n_layers):
        print("iterrr")
        Even(thetaEven_s[i])
        Odd(thetaEven_s[i])
    # during the optimization phase we are evaluating the following expected value:
    return qml.expval(qml.Hermitian(density_matrix(desired_state), wires=[0, 1, 2, 3]))


def QAOA(n_layers=1, verbose=False):
    """ Quantum Approximation Optimization Algorithm:
        Uses a Nesterov Momentum optimizer to variate the hyperparameters theta (one per gate Rx, Rz and layer)
        to minimize the cost 1 - F.
    """
    init_params = 2*np.pi * \
        np.random.rand(
            2, n_layers, n_wires)  # Possible improvement: limit params. to [0, 2pi) (due to periodicity)

    def cost(params):
        thetaEven_s = params[0]
        thetaOdd_s = params[1]
        return (1-np.sqrt(circuit(thetaEven_s, thetaOdd_s, n_layers=n_layers)))

    # itialize optimizer: Nesterov with momentum chosen after trying various
    opt = qml.NesterovMomentumOptimizer(stepsize=0.02, momentum=0.9)

    losses = []

    # optimize parameters in cost
    params = init_params
    for i in range(steps):
        params = opt.step(cost, params)
        loss = cost(params)
        losses.append(loss)
        if verbose == True:
            if (i + 1) % 5 == 0:
                print("Objective after step {:5d}: {: .7f}".format(i + 1, loss))
    print(f"Layer architecture {0} finished".format(n_layers))
    return losses


np.random.seed(42)
desired_state = np.random.random(2**n_wires) + np.random.random(2**n_wires) * 1j
desired_state = desired_state/norm(desired_state)  # normalize

# Execution:
layer_list = [1]


def getLosses(layer_list):
    """
    Returns a list of losses for the different layer setups in layerList.
    """

    losses_list = [QAOA(i) for i in layer_list]
    return losses_list

# losses_list = getLosses(layer_list)
#
# final_losses = [i[-1] for i in losses_list]
#
# for i in range(len(layer_list)):
#     print('Layer : {} | Loss : {}'.format(layer_list[i], final_losses[i]))
