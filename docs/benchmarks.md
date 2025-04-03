# Last stats

```python exec="on" source="material-block" session="benchmarks"
import json # markdown-exec: hide
with open("docs/stats.json", 'r') as f: # markdown-exec: hide
    data= json.load(f)['benchmarks'] # markdown-exec: hide

print(len(data)) # markdown-exec: hide
```
