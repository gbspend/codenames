import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

data = None
with open('results.json') as f:
    data = json.load(f)

l = list(data.keys())
y_names = [name for name in l if 'cheat' not in name]

m = np.zeros((len(y_names), len(l)))

for i,name in enumerate(l):
    if 'cheat' in name:
        continue
    for opp in data[name]:
        j = l.index(opp)
        t = data[name][opp]
        m[i,j] = t[0]/float(sum(t))

mask = np.zeros_like(m)
mask[np.tril_indices_from(mask)] = True

ax = sb.heatmap(m, mask=mask)

plt.draw()
ax.set_xticklabels(l, rotation = 45)
ax.set_yticklabels(y_names, rotation = 0)
plt.tight_layout()

plt.savefig('heat.png')
