from __future__ import annotations

import random
import string
import math
from sympy import Basic
from functools import reduce
from typing import Any, Callable, Set

import hypothesis.strategies as st
from hypothesis.strategies._internal import SearchStrategy

import pyqtorch as pyq
from aquordion.conversion import pyq_to_horqrux_types_match
from aquordion.types import ParameterType, TNumber
from aquordion.utils import extract_parameters


MIN_N_QUBITS = 1
MAX_N_QUBITS = 4
MIN_CIRCUIT_DEPTH = 1
MAX_CIRCUIT_DEPTH = 4

PARAM_NAME_LENGTH = 1

VAR_PARAM_MIN = -2 * math.pi
VAR_PARAM_MAX = 2 * math.pi
PARAM_RANGES = {
    "Variational": (VAR_PARAM_MIN, VAR_PARAM_MAX),
    "Fixed": (VAR_PARAM_MIN, VAR_PARAM_MAX),
}

N_QUBITS_STRATEGY: SearchStrategy[int] = st.integers(min_value=MIN_N_QUBITS, max_value=MAX_N_QUBITS)
CIRCUIT_DEPTH_STRATEGY: SearchStrategy[int] = st.integers(
    min_value=MIN_CIRCUIT_DEPTH, max_value=MAX_CIRCUIT_DEPTH
)

supported_gates: list[pyq.QuantumOperation] = list(pyq_to_horqrux_types_match.keys())
fixed_gateset = [gate for gate in supported_gates if not hasattr(gate, "param_name")]
param_gateset = list(set(supported_gates) - set(fixed_gateset))

def rand_name(length: int) -> str:
    letters = string.ascii_letters
    result_str = "".join(random.choice(letters) for i in range(length))
    return result_str

def get_param(
    param_type: ParameterType,
    name_len: int,
    value: TNumber,
) -> str | TNumber:
    if param_type == "Fixed":
        return value
    else:
        return rand_name(name_len)

# A strategy to generate random parameters.
def rand_parameter(draw: Callable[[SearchStrategy[Any]], Any]) -> Basic:
    param_type = draw(st.sampled_from(list(PARAM_RANGES.keys())))
    min_v, max_v = PARAM_RANGES[param_type]
    value = draw(st.floats(min_value=min_v, max_value=max_v))
    name_len = draw(st.integers(min_value=1, max_value=PARAM_NAME_LENGTH))
    return get_param(draw, param_type=param_type, name_len=name_len, value=value)

# A strategy to generate random blocks.
def rand_digital_blocks(gate_list: list[pyq.QuantumOperation]) -> Callable:
    @st.composite
    def blocks(
        # ops_pool: list[AbstractBlock] TO BE ADDED
        draw: Callable[[SearchStrategy[Any]], Any],
        n_qubits: SearchStrategy[int] = st.integers(min_value=1, max_value=4),
        depth: SearchStrategy[int] = st.integers(min_value=1, max_value=8),
    ) -> pyq.QuantumCircuit:
        
        total_qubits = draw(n_qubits)
        gates_list: list = []
        qubit_indices = {0}

        set_gates_list = set(gate_list).intersection(supported_gates)

        pool_1q = set(gate for gate in set_gates_list if not isinstance(gate, pyq.ControlledPrimitive))


        for _ in range(draw(depth)):
            if total_qubits == 1:
                gate = draw(st.sampled_from(pool_1q))
            else:
                gate = draw(st.sampled_from(set_gates_list))
            
            qubit = draw(st.integers(min_value=0, max_value=total_qubits - 1))
            qubit_indices = qubit_indices.union({qubit})

            if gate in pool_1q:
                if hasattr(gate, "param_name"):
                    gates_list.append(gate(qubit, rand_parameter(draw)))
                else:
                    gates_list.append(gate(qubit))
            else:
                def draw_controls(n: int, except_numbers: tuple[int, ...]) -> tuple[int, ...]:
                    if n == 1:
                        return tuple(set(range(total_qubits)) - set(except_numbers))
                    else:
                        control = st.integers(min_value=0, max_value=total_qubits - 1).filter(
                            lambda x: x not in except_numbers
                        )
                        controls = (control, ) + draw_controls(n - 1, except_numbers + (control, ))
                        return controls

                if isinstance(gate, pyq.ControlledPrimitive):
                    nb_controls = draw(st.integers(min_value=1, max_value=total_qubits - 2))
                    controls = draw_controls(nb_controls, (qubit, ))
                    
                    if hasattr(gate, "param_name"):
                        gates_list.append(gate(target=qubit, control=controls, param_name=rand_parameter(draw)))
                    else:
                        gates_list.append(gate(target=qubit, control=controls))
                else:
                    if hasattr(gate, "param_name"):
                        gates_list.append(gate(qubit, rand_parameter(draw)))
                    else:
                        gates_list.append(gate(qubit))
        return pyq.QuantumCircuit(total_qubits, gates_list)

    return blocks

@st.composite
def digital_circuits(
    draw: Callable[[SearchStrategy[Any]], Any],
    n_qubits: SearchStrategy[int] = N_QUBITS_STRATEGY,
    depth: SearchStrategy[int] = CIRCUIT_DEPTH_STRATEGY,
) -> tuple[pyq.QuantumCircuit, tuple[str, ...]]:
    circuit = draw(rand_digital_blocks(supported_gates)(n_qubits, depth))
    return circuit, extract_parameters(circuit)

