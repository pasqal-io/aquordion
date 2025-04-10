# Stats

We generate time stats using `pytest-benchmark` using $10$ rounds for circuits A, B, C coming from  [^1].
The current execution times are for circuits defined over $2, 5, 10, 15$ qubits and $2, 5$ layers.
So far, we benchmark between `PyQTorch` and `Horqrux` the `run` and `expectation` methods.

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
axes.set_title('Timing distributions by test and circuit')
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
axes.set_title('Timing distributions by test and circuit')
axes.set_xlabel('')
axes.set_ylabel('Time (s)')
axes.set_yscale('log')
plt.xticks(rotation=75)
plt.suptitle('')
plt.tight_layout()
print(fig_to_html(plt.gcf())) # markdown-exec: hide
```

[^1]: [Tyson Jones, Julien Gacon, Efficient calculation of gradients in classical simulations of variational quantum algorithms (2020)](https://arxiv.org/abs/2111.05176)
