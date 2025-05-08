from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
import sympy
import torch
from qadence import Parameter, QuantumCircuit, QuantumModel
from qadence.constructors import total_magnetization
from qadence import HamEvo, X, Y, add, N
from qadence.types import DiffMode
from qadence import Register, QuantumCircuit
from qadence.states import zero_state
from qadence.execution import expectation
from qadence.ml_tools import Trainer, TrainConfig


from qadence import HamEvo, X, Y, N, add
from qadence.analog.constants import C6_DICT
from math import cos, sin

np.random.seed(43)

Trainer.set_use_grad(True)


def create_circuit(n_qubits: int):
    torch.manual_seed(42)
    np.random.seed(43)

    spacing = 7.0
    omega = 5
    detuning = 0
    phase = 0.0

    # differentiable param
    x = Parameter("x", trainable=False)

    # define register
    register = Register.rectangular_lattice(n_qubits, 1, spacing=spacing)


    # Building the terms in the driving Hamiltonian
    h_x = add((omega * (i*0.+1) / 2) * cos(phase) * X(i) for i in range(n_qubits))
    h_y = add((-1.0 * omega * (i*0.+1) / 2) * sin(phase) * Y(i) for i in range(n_qubits))
    h_n = -1.0 * detuning * add(N(i) for i in range(n_qubits))

    # Building the interaction Hamiltonian

    # Dictionary of coefficient values for each Rydberg level, which is 60 by default
    c_6 = C6_DICT[60]
    h_int = c_6 * (
        1/(spacing**6) * (N(1)@N(0))
    )
    for i in range(2, n_qubits):
        for j in range(i):
            s = (i - j) * spacing
            h_int += c_6 * (
                1/(s**6) * (N(i)@N(j))
            )

    hamiltonian = h_x + h_y + h_n + h_int


    # Convert duration to µs due to the units of the Hamiltonian
    block = HamEvo(hamiltonian, x / omega)

    circ = QuantumCircuit(register, block)

    # cost operator
    obs = total_magnetization(n_qubits)

    # initial state
    init_state = zero_state(n_qubits)
    return circ, obs, init_state

system_sizes = [3, 4, ]
gap_step = 3.0
n_eqs = [0, 4, 6, 8, 16]

@pytest.mark.parametrize("n_qubits", system_sizes)
@pytest.mark.parametrize("n_eq", n_eqs)
@pytest.mark.parametrize("order", [2,])
def test_run_psr(benchmark: pytest.Fixture, n_qubits: int, n_eq: int, order: int) -> None:
    circ, obs, init_state = create_circuit(n_qubits)
    config = dict()
    torch.manual_seed(42)
    xs = torch.rand(1, requires_grad=True) * 2 * torch.pi
    values = {"x": xs}

    if n_eq > 0:
        config = {
            "n_eqs": n_eq,
            "gap_step": gap_step,
        }
    if order == 1:
        def eval_grads() -> None:
            expval = expectation(
                circ, state=init_state, observable=obs, values=values, diff_mode=DiffMode.GPSR, configuration=config
            )
            dexpval_x = torch.autograd.grad(
                expval, values["x"], torch.ones_like(expval), create_graph=True
            )[0]
    else:
        def eval_grads() -> None:
            expval = expectation(
                circ, state=init_state, observable=obs, values=values, diff_mode=DiffMode.GPSR, configuration=config
            )
            dexpval_x = torch.autograd.grad(
                expval, values["x"], torch.ones_like(expval), create_graph=True
            )[0]
            ddexpval_xx = torch.autograd.grad(
                dexpval_x, values["x"], torch.ones_like(dexpval_x), create_graph=True
            )[0]
    
    benchmark.pedantic(eval_grads, rounds=10)


# @pytest.mark.parametrize("n_qubits", system_sizes)
# @pytest.mark.parametrize("n_eq", n_eqs)
# def test_vqe_psr(benchmark: pytest.Fixture, n_qubits: int, n_eq: int) -> None:
#     circ, obs, _ = create_circuit(n_qubits)
#     config = dict()
#     if n_eq > 0:
#         config = {
#             "n_eqs": n_eq,
#             "gap_step": gap_step,
#         }
    
#     def train():
#         torch.manual_seed(42)
#         xs = torch.rand(1, requires_grad=True) * 2 * torch.pi
#         values = {"x": xs}
        
#         model = QuantumModel(circ, obs, diff_mode=DiffMode.GPSR, configuration=config)
#         optimizer = torch.optim.Adam(values.values(), lr=1e-3)

#         def loss_fn(model: torch.nn.Module):
#             return torch.mean(model.expectation(values))
    
#         for _ in range(25):
#             optimizer.zero_grad()
#             loss = loss_fn(model)
#             loss.backward()
#             optimizer.step()
    
#     benchmark.pedantic(train, rounds=5)
    
    
