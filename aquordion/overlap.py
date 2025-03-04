from __future__ import annotations

from collections import Counter
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor

from aquordion.blocks.abstract import AbstractBlock
from aquordion.blocks.utils import chain, kron, tag
from aquordion.circuit import QuantumCircuit
from aquordion.divergences import js_divergence
from aquordion.operations import SWAP, H, I, S
from aquordion.transpile import reassign
from aquordion.types import BackendName, OverlapMethod
from aquordion.utils import P0, P1

# Modules to be automatically added to the qadence namespace
__all__: list[str] = []


def _cswap(control: int, target1: int, target2: int) -> AbstractBlock:
    # construct controlled-SWAP block
    cswap_blocks = kron(P0(control), I(target1), I(target2)) + kron(
        P1(control), SWAP(target1, target2)
    )
    cswap = tag(cswap_blocks, f"CSWAP({control}, {target1}, {target2})")

    return cswap


def _controlled_unitary(control: int, unitary_block: AbstractBlock) -> AbstractBlock:
    n_qubits = unitary_block.n_qubits

    # shift qubit support of unitary
    shifted_unitary_block = reassign(unitary_block, {i: control + i + 1 for i in range(n_qubits)})

    # construct controlled-U block
    cu_blocks = kron(P0(control), *[I(control + i + 1) for i in range(n_qubits)]) + kron(
        P1(control), shifted_unitary_block
    )
    cu = tag(cu_blocks, f"c-U({control}, {shifted_unitary_block.qubit_support})")

    return cu


def _is_counter_list(lst: list[Counter]) -> bool:
    return all(map(lambda x: isinstance(x, Counter), lst)) and isinstance(lst, list)


def _select_overlap_method(
    method: OverlapMethod,
    backend: BackendName,
    bra_circuit: QuantumCircuit,
    ket_circuit: QuantumCircuit,
) -> tuple[Callable, QuantumCircuit, QuantumCircuit]:
    if method == OverlapMethod.EXACT:
        fn = overlap_exact

        def _overlap_fn(
            param_values: dict,
            bra_calc_fn: Callable,
            bra_state: Tensor | None,
            ket_calc_fn: Callable,
            ket_state: Tensor | None,
        ) -> Tensor:
            bras = bra_calc_fn(param_values["bra"], bra_state)
            kets = ket_calc_fn(param_values["ket"], ket_state)
            overlap = fn(bras, kets)
            return overlap

    elif method == OverlapMethod.JENSEN_SHANNON:

        def _overlap_fn(
            param_values: dict,
            bra_calc_fn: Callable,
            bra_state: Tensor | None,
            ket_calc_fn: Callable,
            ket_state: Tensor | None,
        ) -> Tensor:
            bras = bra_calc_fn(param_values["bra"], bra_state)
            kets = ket_calc_fn(param_values["ket"], ket_state)
            overlap = overlap_jensen_shannon(bras, kets)
            return overlap

    elif method == OverlapMethod.COMPUTE_UNCOMPUTE:
        # create a single circuit from bra and ket circuits
        bra_circuit = QuantumCircuit(
            bra_circuit.n_qubits, bra_circuit.block, ket_circuit.block.dagger()
        )
        ket_circuit = None  # type: ignore[assignment]

        def _overlap_fn(  # type: ignore [misc]
            param_values: dict, bra_calc_fn: Callable, bra_state: Tensor | None, *_: Any
        ) -> Tensor:
            bras = bra_calc_fn(param_values["bra"], bra_state)
            overlap = overlap_compute_uncompute(bras)
            return overlap

    elif method == OverlapMethod.SWAP_TEST:
        n_qubits = bra_circuit.n_qubits

        # shift qubit support of bra and ket circuit blocks
        shifted_bra_block = reassign(bra_circuit.block, {i: i + 1 for i in range(n_qubits)})
        shifted_ket_block = reassign(
            ket_circuit.block, {i: i + n_qubits + 1 for i in range(n_qubits)}
        )
        ket_circuit = None  # type: ignore[assignment]

        # construct swap test circuit
        state_blocks = kron(shifted_bra_block, shifted_ket_block)
        cswap_blocks = chain(*[_cswap(0, n + 1, n + 1 + n_qubits) for n in range(n_qubits)])
        swap_test_blocks = chain(H(0), state_blocks, cswap_blocks, H(0))
        bra_circuit = QuantumCircuit(2 * n_qubits + 1, swap_test_blocks)

        def _overlap_fn(  # type: ignore [misc]
            param_values: dict, bra_calc_fn: Callable, bra_state: Tensor | None, *_: Any
        ) -> Tensor:
            bras = bra_calc_fn(param_values["bra"], bra_state)
            overlap = overlap_swap_test(bras)
            return overlap

    elif method == OverlapMethod.HADAMARD_TEST:
        n_qubits = bra_circuit.n_qubits

        # construct controlled bra and ket blocks
        c_bra_block = _controlled_unitary(0, bra_circuit.block)
        c_ket_block = _controlled_unitary(0, ket_circuit.block.dagger())

        # construct swap test circuit for Re part
        re_blocks = chain(H(0), c_bra_block, c_ket_block, H(0))
        bra_circuit = QuantumCircuit(n_qubits + 1, re_blocks)

        # construct swap test circuit for Im part
        im_blocks = chain(H(0), c_bra_block, c_ket_block, S(0), H(0))
        ket_circuit = QuantumCircuit(n_qubits + 1, im_blocks)

        def _overlap_fn(
            param_values: dict,
            bra_calc_fn: Callable,
            bra_state: Tensor | None,
            ket_calc_fn: Callable,
            ket_state: Tensor | None,
        ) -> Tensor:
            bras = bra_calc_fn(param_values["bra"], bra_state)
            kets = ket_calc_fn(param_values["ket"], ket_state)
            overlap = overlap_hadamard_test(bras, kets)
            return overlap

    return _overlap_fn, bra_circuit, ket_circuit


def overlap_exact(bras: Tensor, kets: Tensor) -> Tensor:
    """Calculate overlap using exact quantum mechanical definition.

    Args:
        bras (Tensor): full bra wavefunctions
        kets (Tensor): full ket wavefunctions

    Returns:
        Tensor: overlap tensor containing values of overlap of each bra with each ket
    """
    return torch.abs(torch.sum(bras.conj() * kets, dim=1)) ** 2


def fidelity(bras: Tensor, kets: Tensor) -> Tensor:
    return overlap_exact(bras, kets)


def overlap_jensen_shannon(bras: list[Counter], kets: list[Counter]) -> Tensor:
    """Calculate overlap from bitstring counts using Jensen-Shannon divergence method.

    Args:
        bras (list[Counter]): bitstring counts corresponding to bra wavefunctions
        kets (list[Counter]): bitstring counts corresponding to ket wavefunctions

    Returns:
        Tensor: overlap tensor containing values of overlap of each bra with each ket
    """
    return 1 - torch.tensor([js_divergence(p, q) for p, q in zip(bras, kets)])


def overlap_compute_uncompute(bras: Tensor | list[Counter]) -> Tensor:
    """Calculate overlap using compute-uncompute method.

    From full wavefunctions or bitstring counts.

    Args:
        bras (Tensor | list[Counter]): full bra wavefunctions or bitstring counts

    Returns:
        Tensor: overlap tensor containing values of overlap of each bra with zeros ket
    """
    if isinstance(bras, Tensor):
        # calculate exact overlap of full bra wavefunctions with |0> state
        overlap = torch.abs(bras[:, 0]) ** 2

    elif isinstance(bras, list):
        # estimate overlap as the fraction of shots when "0..00" bitstring was observed
        n_qubits = len(list(bras[0].keys())[0])
        n_shots = sum(list(bras[0].values()))
        overlap = torch.tensor([p["0" * n_qubits] / n_shots for p in bras])

    return overlap


def overlap_swap_test(bras: Tensor | list[Counter]) -> Tensor:
    """Calculate overlap using swap test method.

    From full wavefunctions or bitstring counts.

    Args:
        bras (Tensor | list[Counter]): full bra wavefunctions or bitstring counts

    Returns:
        Tensor: overlap tensor
    """
    if isinstance(bras, Tensor):
        n_qubits = int(np.log2(bras.shape[1]))

        # define measurement operator  |0><0| x I
        proj_op = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        ident_op = torch.diag(torch.tensor([1.0 for _ in range(2 ** (n_qubits - 1))]))
        meas_op = torch.kron(proj_op, ident_op).type(torch.complex128)

        # estimate overlap from ancilla qubit measurement
        prob0 = (bras.conj() * torch.matmul(meas_op, bras.t()).t()).sum(dim=1).real

    elif _is_counter_list(bras):
        # estimate overlap as the fraction of shots when 0 was observed on ancilla qubit
        n_qubits = len(list(bras[0].keys())[0])
        n_shots = sum(list(bras[0].values()))
        prob0 = torch.tensor(
            [
                sum(map(lambda k, v: v if k[0] == "0" else 0, p.keys(), p.values())) / n_shots
                for p in bras
            ]
        )
    else:
        raise TypeError("Incorrect type passed for bras argument.")

    # construct final overlap tensor
    overlap = 2 * prob0 - 1

    return overlap


def overlap_hadamard_test(
    bras_re: Tensor | list[Counter], bras_im: Tensor | list[Counter]
) -> Tensor:
    """Calculate overlap using Hadamard test method.

    From full wavefunctions or bitstring counts.

    Args:
        bras_re (Tensor | list[Counter]): full bra wavefunctions or bitstring counts
        for estimation of overlap's real part
        bras_im (Tensor | list[Counter]): full bra wavefunctions or bitstring counts
        for estimation of overlap's imaginary part

    Returns:
        Tensor: overlap tensor
    """
    if isinstance(bras_re, Tensor) and isinstance(bras_im, Tensor):
        n_qubits = int(np.log2(bras_re.shape[1]))

        # define measurement operator  |0><0| x I
        proj_op = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        ident_op = torch.diag(torch.tensor([1.0 for _ in range(2 ** (n_qubits - 1))]))
        meas_op = torch.kron(proj_op, ident_op).type(torch.complex128)

        # estimate overlap from ancilla qubit measurement
        prob0_re = (bras_re * torch.matmul(meas_op, bras_re.conj().t()).t()).sum(dim=1).real
        prob0_im = (bras_im * torch.matmul(meas_op, bras_im.conj().t()).t()).sum(dim=1).real

    elif _is_counter_list(bras_re) and _is_counter_list(bras_im):
        # estimate overlap as the fraction of shots when 0 was observed on ancilla qubit
        n_qubits = len(list(bras_re[0].keys())[0])
        n_shots = sum(list(bras_re[0].values()))
        prob0_re = torch.tensor(
            [
                sum(map(lambda k, v: v if k[0] == "0" else 0, p.keys(), p.values())) / n_shots
                for p in bras_re
            ]
        )
        prob0_im = torch.tensor(
            [
                sum(map(lambda k, v: v if k[0] == "0" else 0, p.keys(), p.values())) / n_shots
                for p in bras_im
            ]
        )
    else:
        raise TypeError("Incorrect types passed for bras_re and kets_re arguments.")

    # construct final overlap tensor
    overlap = (2 * prob0_re - 1) ** 2 + (2 * prob0_im - 1) ** 2

    return overlap
