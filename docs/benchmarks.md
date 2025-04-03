# Last stats

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
frame['fn_circuit'] = frame['fn_circuit'].apply(lambda x: re.findall('function (.*) at', x)[0]) # markdown-exec: hide

```

## Run method

```python exec="on" source="material-block" session="benchmarks"

run_frame = frame[frame['name'].str.startswith('run')] # markdown-exec: hide
axes = frame.boxplot('median', by=['name', 'fn_circuit']) # markdown-exec: hide
axes.set_title('Timing distributions by test and circuit') # markdown-exec: hide
axes.set_xlabel('') # markdown-exec: hide
axes.set_ylabel('Time (s)') # markdown-exec: hide
axes.set_yscale('log') # markdown-exec: hide
plt.xticks(rotation=75) # markdown-exec: hide
plt.suptitle('') # markdown-exec: hide
plt.show() # markdown-exec: hide

```
