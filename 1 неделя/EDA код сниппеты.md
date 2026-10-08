---

---
# EDA код сниппеты

```python
dfnull = pd.DataFrame({
    0: np.random.permutation([np.nan]*30 + list(range(70))),
    1: np.random.permutation([np.nan]*50 + list(range(50))),
})

plt.figure(figsize=(20,10))

cmap = sns.cubehelix_palette(as_cmap=True, light=.9)

sns.heatmap(dfnull.isna().transpose(), cmap=cmap,
            cbar_kws={'label': 'Missing Data'})
```


![422](attachments/Pasted%20image%2020260910120751.png)


```python
d = df.assign(val_q=pd.qcut(df[TARGET], 4, labels=["Q1", "Q2", "Q3", "Q4"]))
sns.catplot(data=d, x="val_q", y="AveOccup", kind="boxen", height=6, aspect=2)
plt.ylim(0, 10)
```
![458](attachments/Pasted%20image%2020260912105427.png)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

d = df.assign(lat_bin=pd.cut(df["Latitude"], bins=10))
order = [str(c) for c in reversed(d["lat_bin"].cat.categories)]  

pal = sns.color_palette("magma", n_colors=len(order))

sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

g = sns.FacetGrid(d, row="lat_bin", row_order=order, hue="lat_bin", hue_order=order,
                  aspect=10, height=0.85, palette=pal)

g.map(sns.kdeplot, TARGET, bw_adjust=0.5, clip_on=False, fill=True, alpha=0.85, lw=0)
g.map(sns.kdeplot, TARGET, bw_adjust=0.5, clip_on=False, color="w", lw=1.4)
g.refline(y=0, lw=1.4, ls="-", color=None, clip_on=False)

def label(x, color, label):
    plt.gca().text(0, .25, label, fontweight="bold", color=color, fontsize=9,
                   ha="left", va="center", transform=plt.gca().transAxes)

g.map(label, TARGET)
g.figure.subplots_adjust(hspace=-0.55)         
g.set_titles("")
g.set(yticks=[], ylabel="", xlabel="MedHouseVal")
g.despine(bottom=True, left=True)
g.figure.suptitle("Распределение MedHouseVal по широтным полосам", y=0.98)

sns.set_theme(style="whitegrid")               
```
![455](attachments/Pasted%20image%2020260912105646.png)
```python
import numpy as np
import matplotlib.pyplot as plt


def compare_hists(df, group_col, labels, columns=None, skip=(),
                  nrows=4, ncols=4, figsize=(22, 16), title=None):
    columns = columns or [c for c in df.columns if c != group_col]
    lab_a, lab_b = labels

    with plt.rc_context({'font.size': 7}):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten()
        fig.suptitle(title or f"Сравнение распределений признаков: {lab_a} vs {lab_b}", fontsize=9)
        plt.subplots_adjust(top=0.93)

        for i, feat in enumerate(columns):
            axes[i].hist(df[df[group_col] == lab_a][feat], bins=10, alpha=0.5, color='blue', label=lab_a, density=True)
            axes[i].hist(df[df[group_col] == lab_b][feat], bins=10, alpha=0.5, color='orange', label=lab_b, density=True)
            axes[i].set_title(feat, fontsize=8)

            if feat not in skip:
                mean_a = df[df[group_col] == lab_a][feat].mean()
                mean_b = df[df[group_col] == lab_b][feat].mean()
                axes[i].axvline(mean_a, c="blue", linestyle='--', linewidth=1, label=f'{lab_a} mean: {mean_a:.2f}')
                axes[i].axvline(mean_b, c="orange", linestyle='--', linewidth=1, label=f'{lab_b} mean: {mean_b:.2f}')

            axes[i].legend(fontsize=6)

        for i in range(len(columns), len(axes)):
            fig.delaxes(axes[i])

        plt.show()


med = df[TARGET].median()
df["Price"] = np.where(df[TARGET] >= med, "High", "Low")
compare_hists(df, "Price", ("High", "Low"), skip=("Price",))
```

![](attachments/Pasted%20image%2020260912111428.png)

```python
sns.clustermap(df.corr(method="spearman"), cmap="vlag", center=0,
               figsize=(9, 9), linewidths=.4, annot=True, fmt=".2f")
```
![422](attachments/Pasted%20image%2020260912105452.png)
```python
sns.jointplot(data=df, x="MedInc", y=TARGET, kind="hex",
              height=8, marginal_kws={"bins": 50})
```
![331](attachments/Pasted%20image%2020260912105516.png)

```python
df.plot(kind="scatter", x="Longitude", y="Latitude", alpha=0.4,
        s=df["Population"] / 100, label="population", figsize=(10, 7),
        c=TARGET, cmap=plt.get_cmap("jet"), colorbar=True, sharex=False)
plt.legend()
```
![452](attachments/Pasted%20image%2020260912105541.png)
```python
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

STYLE = {"mean": "red", "gmean": "green", "median": "blue", "mode": "olive",
         "q1": "orange", "q3": "orange", "d1": "brown", "d9": "brown"}


def describe_stats(s):
    out = s.quantile([.1, .25, .5, .75, .9])
    out.index = ["d1", "q1", "median", "q3", "d9"]
    out["mean"] = s.mean()
    out["gmean"] = stats.gmean(s[s > 0])
    out["mode"] = s.mode().iat[0]
    return out


def hist_box(df, col, bins="auto", log=False, figsize=(15, 5)):
    s = df[col].dropna()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize,
                                   gridspec_kw={"width_ratios": [3, 1]})
    sns.histplot(s, kde=True, bins=bins, color="skyblue", ax=ax1, log_scale=log)
    sns.boxplot(y=s, ax=ax2, width=.4, color="skyblue")
    if log:
        ax2.set_yscale("log")

    for name, val in describe_stats(s).items():
        ax1.axvline(val, color=STYLE[name], ls="--", lw=1.2, label=f"{name} = {val:.2f}")
        ax2.axhline(val, color=STYLE[name], ls="--", lw=1.2)

    ax1.set(title=f"Распределение {col}", xlabel=col, ylabel="Частота")
    ax1.legend(fontsize=8)
    ax2.set(title="Боксплот", ylabel="")
    fig.tight_layout()
    plt.show()


hist_box(df, "MedInc")
```
![](attachments/Pasted%20image%2020260912111507.png)