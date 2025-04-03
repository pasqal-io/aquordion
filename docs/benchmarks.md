# Last stats

```python exec="on" source="material-block" session="benchmarks"
import json # markdown-exec: hide
with open("stats.json", 'r') as f: # markdown-exec: hide
    data= json.load(f)['benchmarks'] # markdown-exec: hide

data_stats = [{'name': x['name']} | x['params'] | x['stats'] for x in data] # markdown-exec: hide

frame = pd.DataFrame(data_stats) # markdown-exec: hide
frame['name'] = frame['name'].apply(lambda x: re.findall('test_(.*)\\[', x)[0]) # markdown-exec: hide
frame['fn_circuit'] = frame['fn_circuit'].apply(lambda x: re.findall('function (.*) at', x)[0]) # markdown-exec: hide

```

## Run method

```python exec="on" source="material-block" session="benchmarks"
run_frame = frame[frame['name'].str.startswith('run')]
axes = frame.boxplot('median', by=['name', 'fn_circuit'])
axes.set_title('Timing distributions by test and circuit')
axes.set_xlabel('')
axes.set_ylabel('Time (s)')
axes.set_yscale('log')
plt.xticks(rotation=75)
plt.suptitle('')
plt.show()
```
