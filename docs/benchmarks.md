# Stats

We generate time stats using `pytest-benchmark` using $10$ rounds for circuits A, B, C coming from  [^1].
The current execution times are for circuits defined over $2, 5, 10, 15$ qubits and $2, 5$ layers.
So far, we benchmark between `PyQTorch` and `Horqrux` the `run` and `expectation` methods.

```python exec="on" source="material-block" session="benchmarks"

import json # markdown-exec: hide
import pandas as pd # markdown-exec: hide
import re # markdown-exec: hide
import matplotlib.pyplot as plt # markdown-exec: hide

with open("stats.json", 'r') as f: # markdown-exec: hide
    data= json.load(f)['benchmarks'] # markdown-exec: hide

data_stats = [{'name': x['name']} | x['params'] | x['stats'] for x in data] # markdown-exec: hide

frame = pd.DataFrame(data_stats) # markdown-exec: hide
frame['name'] = frame['name'].apply(lambda x: re.findall('test_(.*)\\[', x)[0]) # markdown-exec: hide
frame['fn_circuit'] = frame['benchmark_circuit'].apply(str)
frame['fn_circuit'] = frame['fn_circuit'].apply(lambda x: re.findall('function (.*) at', x)[0]) # markdown-exec: hide

```

## Run method

Here are the median execution times for the `run` method over a random state.

```python exec="on" source="material-block" session="benchmarks"

run_frame = frame[frame['name'].str.startswith('run')] # markdown-exec: hide
run_frame['name'] = run_frame['name'].str.replace('run_', '') # markdown-exec: hide

axes = frame.boxplot('median', by=['name', 'fn_circuit']) # markdown-exec: hide
axes.set_title('Timing distributions by test and circuit') # markdown-exec: hide
axes.set_xlabel('') # markdown-exec: hide
axes.set_ylabel('Time (s)') # markdown-exec: hide
axes.set_yscale('log') # markdown-exec: hide
plt.xticks(rotation=75) # markdown-exec: hide
plt.suptitle('') # markdown-exec: hide
plt.show() # markdown-exec: hide

```

## Expectation method: Z(0) observable

Here are the median execution times for the `expectation` method over a random state and the $Z(0)$ observable.

```python exec="on" source="material-block" session="benchmarks"

expectation_frame = frame[frame['name'].str.startswith('expectation')] # markdown-exec: hide
expectation_frame['name'] = expectation_frame['name'].str.replace('expectation_', '') # markdown-exec: hide
axes = frame.boxplot('median', by=['name', 'fn_circuit']) # markdown-exec: hide
axes.set_title('Timing distributions by test and circuit') # markdown-exec: hide
axes.set_xlabel('') # markdown-exec: hide
axes.set_ylabel('Time (s)') # markdown-exec: hide
axes.set_yscale('log') # markdown-exec: hide
plt.xticks(rotation=75) # markdown-exec: hide
plt.suptitle('') # markdown-exec: hide
plt.show() # markdown-exec: hide

```

[^1]: [Tyson Jones, Julien Gacon, Efficient calculation of gradients in classical simulations of variational quantum algorithms (2020)](https://arxiv.org/abs/2111.05176)
