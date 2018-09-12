from pyquil.quil import Program
import pyquil.api as api
from pyquil.gates import *
from pyquil.api import QVMConnection
import numpy as np
from pyquil.api import get_devices
import argparse

# Helper function: Applies hadamard to the first n registers in program p
def hadamard_n(n, p):
    for i in range(n):
        p.inst(H(i))

# Helper function: Builds Uf unitary, which essentially applies the identity
# matrix to all inputs except input s, which gets an X applied to it
def build_uf(n, s):
    uf = np.identity(2**(n+1))          # Create the identity
    uf[2*s, 2*s] = 0.0                  # Create an X to make s input result
    uf[2*s, 2*s+1] = 1.0                #   in a flip
    uf[2*s+1, 2*s] = 1.0
    uf[2*s+1, 2*s+1] = 0.0
    return uf

# Helper function: concatenates a string of n(+1) registers in the correct format
# for applying unitaries to multiple registers
def build_gate_string(n):
    gates = ""
    for i in range(n+1):
        gates += " " + str(i)
    return gates

# Helper function: builds the G unitary in Grovers algorithm by taking 2 times
# the ket-bra of the zero ket tensor'd with itself n times, and subtracting the
# identity from that. This matrix essentially ends up being a positive 1 in the
# top left, and negative ones on the diagonal (zeros everywhere else)
def build_ug(n):
    zero_ket = np.array([1.0, 0.0])                     # build the zero ket
    zero_ketbra = np.outer(zero_ket, zero_ket)          # take the ket-bra
    zero_ketbra_build = np.outer(zero_ket, zero_ket)    # this is a placeholder
    identity = np.array([[1.0, 0.0],                    # build the identity
                         [0.0, 1.0]])
    identity_build = np.array([[1.0, 0.0],              # identity placeholder
                               [0.0, 1.0]])
    for i in range(n-1):
        zero_ketbra_build = np.kron(zero_ketbra_build, zero_ketbra)  #tensor
        identity_build = np.kron(identity_build, identity)

    return np.subtract(2*zero_ketbra_build, identity_build)



# This function sets up and creates a program that can run Grover's Algorithm.
# Argument "n" denotes the number of qubits/the length of the function inputs,
# and argument "s" denotes the single input that maps to 1.
def grover(n, s):
    p = Program()               # create program

    ##################
    ## Define Gates ##
    ##################
    uf = build_uf(n, s)         # build Uf
    p.defgate("Uf", uf)         # define Uf

    ug = build_ug(n)            # build Ug
    p.defgate("Ug", ug)         # define Ug

    gate_string = build_gate_string(n)      # useful for applying gates


    ####################
    ## Create Program ##
    ####################
    hadamard_n(n, p)           # apply Hadamard to first N registers
    p.inst(X(n), H(n))         # apply X and N to last register to get minus

    for i in range(int((np.pi*(2**(n/2)))/4)):   # repeat (pi*2^(n/2))/4 times
        p.inst(("Uf"+gate_string))               # apply Uf to all registers
        hadamard_n(n, p)                         # apply Hadamard to first N
        p.inst(("Ug"+gate_string[:-2]))          # apply Ug to first N
        hadamard_n(n, p)                         # apply Hadamard to first N

    for i in range(n):
        p.measure(i, i)        # measure first N qubits, store in same registers

    return p



####################################
# Run a program from command line: #
#    Ex: python3 grover.py 6 32    #
#        out: [[1, 0, 0, 0, 0, 0]] #
####################################
qvm = QVMConnection()
parser = argparse.ArgumentParser()
parser.add_argument("n", help="the number of qubits/length of a given input", type=int)
parser.add_argument("s", help="the input that gets mapped to 1", type=int)
args = parser.parse_args()
n = args.n
s = args.s
p = grover(n, s)
print(qvm.run(p, range(n)))







# Sample Output (first arg is N, second is S):
#    i.e. of form python3 grover.py n s

# peterbromley$ python3 grover.py 2 3
# [[1, 1]]
# peterbromley$ python3 grover.py 2 1
# [[0, 1]]
# peterbromley$ python3 grover.py 3 7
# [[1, 1, 1]]
# peterbromley$ python3 grover.py 4 10
# [[1, 0, 1, 0]]
# peterbromley$ python3 grover.py 5 10
# [[0, 1, 0, 1, 0]]
# peterbromley$ python3 grover.py 5 20
# [[1, 0, 1, 0, 0]]
# peterbromley$ python3 grover.py 6 63
# [[1, 1, 1, 1, 1, 1]]
# peterbromley$ python3 grover.py 6 1
# [[0, 0, 0, 0, 0, 1]]
# peterbromley$ python3 grover.py 6 32
# [[1, 0, 0, 0, 0, 0]]
# peterbromley$ python3 grover.py 8 255
# [[1, 1, 1, 1, 1, 1, 1, 1]]
