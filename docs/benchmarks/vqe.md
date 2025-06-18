# Stats

We generate timing statistics using `pytest-benchmark` using $R$ rounds for circuits A, B, C [^1] as ansatze for a variational quantum eigensolver task[^2] (VQE) for the $H2$ molecule in the STO-3G basis with a bondlength of $0.742 \mathring{A}$[^3].
The underlying gradient-based Adam optimizer is run for $50$ iterations without shots and $10$ when using shots.
The circuits are defined over $4, 6$ qubits $R=5$ for avoiding long jobs time on Github.
Additionally, we benchmark two differentiation modes (automatic differentiation and the Adjoint method [^1]).
Finally, we also performing shot-based benchmarks, we use $100$ shots.

Note we are disabling shots due to an issue.

# Variational Quantum Eigensolver

Here are the median execution times for VQE. We compare optimizing with `PyQTorch` against optimizing with `Horqrux` and jitting.

```python exec="on" source="material-block" session="benchmarks"
import json
import pandas as pd
import re
import matplotlib.pyplot as plt

import os.path

fname = "stats_vqe_noshots.json"
fnameshots = "stats_vqe_shots.json"

if not os.path.isfile(fname):
    fname = "docs/stats_vqe_noshots.json"
    fnameshots = "docs/stats_vqe_shots.json"
with open(fname, 'r') as f:
    data_vqe = json.load(f)['benchmarks']
with open(fnameshots, 'r') as f:
    data_vqeshots = json.load(f)['benchmarks']

data_stats_vqe = [{'name': x['name']} | x['params'] | x['stats'] for x in data_vqe]
data_stats_vqeshots = [{'name': x['name']} | x['params'] | x['stats'] for x in data_vqeshots]

frame_vqe = pd.DataFrame(data_stats_vqe)
frame_vqe['n_qubits'] = frame_vqe['name'].apply(lambda x: int(re.findall('n:(.*)\\D:', x)[0]))
frame_vqe['name'] = frame_vqe['name'].apply(lambda x: re.findall('test_(.*)\\[', x)[0])
frame_vqe['fn_circuit'] = frame_vqe['benchmark_vqe_ansatz'].apply(str)
frame_vqe['fn_circuit'] = frame_vqe['fn_circuit'].apply(lambda x: re.findall('function (.*) at', x)[0])
frame_vqe['name'] = frame_vqe['name'].str.replace('vqe_', '')
frame_vqe['n_shots'] = 0

frame_vqeshots = pd.DataFrame(data_stats_vqeshots)
frame_vqeshots['n_qubits'] = frame_vqeshots['name'].apply(lambda x: int(re.findall('n:(.*)\\D:', x)[0]))
frame_vqeshots['name'] = frame_vqeshots['name'].apply(lambda x: re.findall('test_(.*)\\[', x)[0])
frame_vqeshots['fn_circuit'] = frame_vqeshots['benchmark_vqe_ansatz'].apply(str)
frame_vqeshots['fn_circuit'] = frame_vqeshots['fn_circuit'].apply(lambda x: re.findall('function (.*) at', x)[0])
frame_vqeshots['name'] = frame_vqeshots['name'].str.replace('vqe_', '')
frame_vqeshots['diff_mode'] = 'ad'
frame_vqeshots['n_shots'] = 100

nqubits = frame_vqe.n_qubits.unique()
```

## Timings

Below we present the distribution of median times for each circuit type, with and without shots.

```python exec="on" source="material-block" session="benchmarks"
for nq in nqubits:
    axes = frame_vqe[frame_vqe.n_qubits == nq].boxplot('median', by=['fn_circuit', 'name', 'diff_mode'])
    axes.set_title(f"Timing distributions by differentiation methods and circuit \n without shots - 100 epochs - {nq} qubits")
    axes.set_xlabel('')
    axes.set_ylabel('Time (s)')
    plt.xticks(rotation=75)
    plt.suptitle('')
    plt.tight_layout()
    print(fig_to_html(plt.gcf())) # markdown-exec: hide



    axes = frame_vqeshots[frame_vqeshots.n_qubits == nq].boxplot('median', by=['fn_circuit', 'name'])
    axes.set_title(f"Timing distributions by differentiation methods and circuit \n with shots - 50 epochs - {nq} qubits")
    axes.set_xlabel('')
    axes.set_ylabel('Time (s)')
    plt.xticks(rotation=75)
    plt.suptitle('')
    plt.tight_layout()
    print(fig_to_html(plt.gcf())) # markdown-exec: hide

```

## Speed-ups

Below we present the distribution of median speed-ups for each circuit type. The timing ratio between `PyQTorch` (numerator) and `Horqrux` (denominator) executions is computed. A ratio higher than $1$ means `Horqrux` and jitting provide computational speed-up over `PyQTorch`.

```python exec="on" source="material-block" session="benchmarks"

frame_vqe = pd.concat([frame_vqe, frame_vqeshots], ignore_index=True)

pyq_vqe = frame_vqe[frame_vqe.name == 'pyq'][['fn_circuit', 'median', 'n_shots', 'diff_mode', 'n_qubits']]
horqrux_vqe = frame_vqe[frame_vqe.name == 'horqrux'][['fn_circuit', 'median', 'n_shots', 'diff_mode', 'n_qubits']]
ratio_df = pd.merge(pyq_vqe, horqrux_vqe, on=['fn_circuit', 'diff_mode', 'n_shots', 'n_qubits'], suffixes=['_pyq', '_horqrux'])
ratio_df['ratio'] = ratio_df['median_pyq'] / ratio_df['median_horqrux']


axes = ratio_df[ratio_df.n_shots == 0].boxplot('ratio', by=['fn_circuit', 'diff_mode', 'n_qubits'])
axes.set_title(f"Speedup distributions by circuit and qubit number without shots \n 100 epochs ")
axes.set_xlabel('')
axes.set_ylabel('Speedup')
plt.xticks(rotation=75)
plt.suptitle('')
plt.tight_layout()
print(fig_to_html(plt.gcf())) # markdown-exec: hide


axes = ratio_df[ratio_df.n_shots > 0].boxplot('ratio', by=['fn_circuit', 'n_qubits'])
axes.set_title(f"Speedup distributions by circuit and qubit number with shots \n 50 epochs")
axes.set_xlabel('')
axes.set_ylabel('Speedup')
plt.xticks(rotation=75)
plt.suptitle('')
plt.tight_layout()
print(fig_to_html(plt.gcf())) # markdown-exec: hide

```


[^1]: [Tyson Jones, Julien Gacon, Efficient calculation of gradients in classical simulations of variational quantum algorithms (2020)](https://arxiv.org/abs/2111.05176)
[^2]: [Tilly et al., The Variational Quantum Eigensolver: a review of methods and best practices (2022)](https://arxiv.org/abs/2111.05176)
[^3]: [Pennylane, Quantum Datasets](https://docs.pennylane.ai/en/stable/introduction/data.html)
