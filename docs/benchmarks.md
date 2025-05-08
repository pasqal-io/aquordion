# Stats

We generate timing statistics using `pytest-benchmark` using $R$ rounds for circuits A, B, C [^1].
So far, we benchmark between `PyQTorch` and `Horqrux`:

- the `run` method,
- the `expectation` method using a single observable Z,
- a variational quantum eigensolver[^2] (VQE) for the $H2$ molecule in the STO-3G basis with a bondlength of $0.742 \mathring{A}$[^3]. The underlying gradient-based Adam optimizer is run for $50$ iterations.

The current execution times (with $R=10$) are for circuits defined over $2, 5, 10, 15$ qubits and $2, 5$ layers for the `run` and `expectation` methods.
For VQE, we reduce the tests to only $10$ qubits for avoiding long jobs time on Github and $R=5$,
and we also benchmark two differentiation modes (automatic differentiation and the Adjoint method [^1]).
Additionally, when using $1000$ shots, we reduce the number of iterations to $10$.


```python exec="on" source="material-block" session="benchmarks"

import json
import pandas as pd
import re
import matplotlib.pyplot as plt

import os.path
fname = "stats.json"
if not os.path.isfile(fname):
    fname = "docs/stats.json"
with open(fname, 'r') as f:
    data= json.load(f)['benchmarks']

data_stats = [{'name': x['name']} | x['params'] | x['stats'] for x in data]

frame = pd.DataFrame(data_stats)
frame['name'] = frame['name'].apply(lambda x: re.findall('test_(.*)\\[', x)[0])
frame['fn_circuit'] = frame['benchmark_circuit'].apply(str)
frame['fn_circuit'] = frame['fn_circuit'].apply(lambda x: re.findall('function (.*) at', x)[0])

```

## Run method

Here are the median execution times for the `run` method over a random state.

```python exec="on" source="material-block" session="benchmarks"

run_frame = frame[frame['name'].str.startswith('run')]
run_frame['name'] = run_frame['name'].str.replace('run_', '')

axes = run_frame.boxplot('median', by=['fn_circuit', 'name'])
axes.set_title("Timing distributions by test and circuit \n for `run` method")
axes.set_xlabel('')
axes.set_ylabel('Time (s)')
axes.set_yscale('log')
plt.xticks(rotation=75)
plt.suptitle('')
plt.tight_layout()
from io import StringIO  # markdown-exec: hide
from matplotlib.figure import Figure  # markdown-exec: hide
def fig_to_html(fig: Figure) -> str:  # markdown-exec: hide
    buffer = StringIO()  # markdown-exec: hide
    fig.savefig(buffer, format="svg")  # markdown-exec: hide
    return buffer.getvalue()  # markdown-exec: hide
# from docs import docutils # markdown-exec: hide
print(fig_to_html(plt.gcf())) # markdown-exec: hide
```

## Expectation method: Z(0) observable

Here are the median execution times for the `expectation` method over a random state and the $Z(0)$ observable.

```python exec="on" source="material-block" session="benchmarks"

expectation_frame = frame[frame['name'].str.startswith('expectation')]
expectation_frame['name'] = expectation_frame['name'].str.replace('expectation_', '')
axes = expectation_frame.boxplot('median', by=['fn_circuit', 'name'])
axes.set_title("Timing distributions by test and circuit \n for `expectation` method")
axes.set_xlabel('')
axes.set_ylabel('Time (s)')
axes.set_yscale('log')
plt.xticks(rotation=75)
plt.suptitle('')
plt.tight_layout()
print(fig_to_html(plt.gcf())) # markdown-exec: hide
```

## VQE

Here are the median execution times for VQE. We compare optimizing with `PyQTorch` against optimizing with `Horqrux` and jitting.

```python exec="on" source="material-block" session="benchmarks"
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
frame_vqe['name'] = frame_vqe['name'].apply(lambda x: re.findall('test_(.*)\\[', x)[0])
frame_vqe['fn_circuit'] = frame_vqe['benchmark_vqe_ansatz'].apply(str)
frame_vqe['fn_circuit'] = frame_vqe['fn_circuit'].apply(lambda x: re.findall('function (.*) at', x)[0])
frame_vqe['name'] = frame_vqe['name'].str.replace('vqe_', '')
frame_vqe['n_shots'] = 0

frame_vqeshots = pd.DataFrame(data_stats_vqeshots)
frame_vqeshots['name'] = frame_vqeshots['name'].apply(lambda x: re.findall('test_(.*)\\[', x)[0])
frame_vqeshots['fn_circuit'] = frame_vqeshots['benchmark_vqe_ansatz'].apply(str)
frame_vqeshots['fn_circuit'] = frame_vqeshots['fn_circuit'].apply(lambda x: re.findall('function (.*) at', x)[0])
frame_vqeshots['name'] = frame_vqeshots['name'].str.replace('vqe_', '')
frame_vqeshots['n_shots'] = 1000
frame_vqeshots['diff_mode'] = 'ad'
```

### Times

Below we present the distribution of median times for each circuit type, without shots and with shots ($n_shots = 1000$).

```python exec="on" source="material-block" session="benchmarks"
axes = frame_vqe.boxplot('median', by=['fn_circuit', 'name', 'diff_mode'])
axes.set_title('Timing distributions by test and circuit without shots - 50 epochs')
axes.set_xlabel('')
axes.set_ylabel('Time (s)')
#axes.set_yscale('log')
plt.xticks(rotation=75)
plt.suptitle('')
plt.tight_layout()
print(fig_to_html(plt.gcf())) # markdown-exec: hide



axes = frame_vqeshots.boxplot('median', by=['fn_circuit', 'name'])
axes.set_title('Timing distributions by test and circuit with shots  - 10 epochs')
axes.set_xlabel('')
axes.set_ylabel('Time (s)')
#axes.set_yscale('log')
plt.xticks(rotation=75)
plt.suptitle('')
plt.tight_layout()
print(fig_to_html(plt.gcf())) # markdown-exec: hide

```

### Speed-ups

Below we present the distribution of median speed-ups for each circuit type. The timing ratio between `PyQTorch` (numerator) and `Horqrux` (denominator) executions is computed. A ratio higher than $1$ means `Horqrux` and jitting provide computational speed-up over `PyQTorch`.

```python exec="on" source="material-block" session="benchmarks"

frame_vqe = pd.concat([frame_vqe, frame_vqeshots], ignore_index=True)

pyq_vqe = frame_vqe[frame_vqe.name == 'pyq'][['fn_circuit', 'median', 'n_shots', 'diff_mode']]
horqrux_vqe = frame_vqe[frame_vqe.name == 'horqrux'][['fn_circuit', 'median', 'n_shots', 'diff_mode']]
ratio_df = pd.merge(pyq_vqe, horqrux_vqe, on=['fn_circuit', 'diff_mode', 'n_shots'], suffixes=['_pyq', '_horqrux'])
ratio_df['ratio'] = ratio_df['median_pyq'] / ratio_df['median_horqrux']
axes = ratio_df[ratio_df.n_shots == 0].boxplot('ratio', by=['fn_circuit', 'diff_mode'])
axes.set_title('Speedup distributions by circuit without shots - 50 epochs')
axes.set_xlabel('')
axes.set_ylabel('Speedup')
plt.xticks(rotation=75)
plt.suptitle('')
plt.tight_layout()
print(fig_to_html(plt.gcf())) # markdown-exec: hide


axes = ratio_df[ratio_df.n_shots > 0].boxplot('ratio', by='fn_circuit')
axes.set_title('Speedup distributions by circuit with shots - 10 epochs')
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
