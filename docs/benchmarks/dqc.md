# Stats

In this section, we benchmark between backends to solve a partial differential equation using a Differential Quantum Circuit ([DQC](https://arxiv.org/abs/2011.10395)). The underlying ansatz is the hardware-efficient-ansatz.
The underlying gradient-based Adam optimizer is run for $25$ iterations.
The example is taken from the `pyqtorch` and `horqrux` documentation.
The circuits are defined over $4, 10$ qubits $R=5$ for avoiding long jobs time on Github.


# Differential Quantum Circuit

Here are the median execution times for DQC. We compare optimizing with `PyQTorch` against optimizing with `Horqrux` and jitting.

```python exec="on" source="material-block" session="benchmarks"

fname = "stats_dqc.json"

if not os.path.isfile(fname):
    fname = "docs/stats_dqc.json"
with open(fname, 'r') as f:
    data_vqe = json.load(f)['benchmarks']


data_stats_vqe = [{'name': x['name']} | x['params'] | x['stats'] for x in data_vqe]

frame_vqe = pd.DataFrame(data_stats_vqe)
frame_vqe['n_qubits'] = frame_vqe['name'].apply(lambda x: int(re.findall('n:(.*)\\D:', x)[0]))
frame_vqe['name'] = frame_vqe['name'].apply(lambda x: re.findall('test_(.*)\\[', x)[0])
frame_vqe['fn_circuit'] = frame_vqe['benchmark_dqc_ansatz'].apply(str)
frame_vqe['fn_circuit'] = frame_vqe['fn_circuit'].apply(lambda x: re.findall('function (.*) at', x)[0])
frame_vqe['name'] = frame_vqe['name'].str.replace('dqc_', '')

nqubits = frame_vqe.n_qubits.unique()
```

## Timings

Below we present the distribution of median times for each number of qubits.

```python exec="on" source="material-block" session="benchmarks"
for nq in nqubits:
    axes = frame_vqe[frame_vqe.n_qubits == nq].boxplot('median', by=['fn_circuit', 'name'])
    axes.set_title(f"Timing distributions - {nq} qubits")
    axes.set_xlabel('')
    axes.set_ylabel('Time (s)')
    plt.xticks(rotation=75)
    plt.suptitle('')
    plt.tight_layout()
    print(fig_to_html(plt.gcf())) # markdown-exec: hide
```

## Speed-ups

```python exec="on" source="material-block" session="benchmarks"
pyq_vqe = frame_vqe[frame_vqe.name == 'pyq'][['fn_circuit', 'median', 'n_qubits']]
horqrux_vqe = frame_vqe[frame_vqe.name == 'horqrux'][['fn_circuit', 'median', 'n_qubits']]
ratio_df = pd.merge(pyq_vqe, horqrux_vqe, on=['fn_circuit', 'n_qubits'], suffixes=['_pyq', '_horqrux'])
ratio_df['ratio'] = ratio_df['median_pyq'] / ratio_df['median_horqrux']


axes = ratio_df.boxplot('ratio', by=['fn_circuit', 'n_qubits'])
axes.set_title(f"Speedup distributions by circuit and qubit number without shots")
axes.set_xlabel('')
axes.set_ylabel('Speedup')
plt.xticks(rotation=75)
plt.suptitle('')
print(fig_to_html(plt.gcf())) # markdown-exec: hide
```
