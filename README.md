# Authorship Clustering

**Rank List Fusion using multiple Text Representations Strategies for Authorship Clustering.**

This repository contains files related to Margueron's *Authorship clustering* master thesis (2021).

## Abstract

This study aim to select different methods, text representations and distance metrics to implement and evaluate different strategies to solve the authorship clustering problem.
In this case, having a set of $n$ texts written by an unknown number of authors, the task is to regroup under the same cluster all texts written by the same author.
The number of different author denoted by $k$ can variate from 1 (all texts are written by the same author) to $n$ (each text is written by a distinct author).

To represent the underlying style of each document various text representation have been suggested: words frequencies, lemmas frequencies, letters $n$-grams frequencies or part-of-speech (POS) sequences frequencies.
Each text is then represented by a vector with a number of dimension corresponding to the number of features.
$L^1$, $L^2$ and inner product based distances have been proposed in conjunction to compute distance between vectors representations.
In addition to the vector representation, compression techniques are also applied.
The Oxquarry, Brunet and St-Jean corpora are used to evaluate the system effectiveness.

In a first part, the rank lists obtained by each individual representation are used to solve the authorship clustering task.
From a rank list, an automatic clustering algorithm can generate the corresponding clusters, hopefully regrouping all the texts written by a given author under the same cluster.
The clustering task is achieved by using three different models based on hierarchical clustering.
A Silhouette score based model, a distribution-based model and a regression-based clustering models are proposed and evaluated.
The clustering obtained by these models are close to the best achievable clustering for the rank lists used.
In a second part, a new rank list is created by fusing rank lists obtained using different strategies.
Two approaches are explored for the fusion: one use Z-Score normalization and another one use logistic regression.
The rank list obtained by fusion is shown to have better performances than the best individual rank list.
In addition to the fusion, two veto strategy are proposed to try to enhance the fusion quality.
The veto does not show any significant improvements with the rank lists used.
Lastly, we showed that using rank lists obtained by fusion yield better clustering results than the ones obtained by individual methods.

## Thesis / Report

[Authorship Clustering thesis](MARGUERON_Authorship_Clustering.pdf)

## Repository organization

```
.
├── README.md              | This document
├── .githooks              | Hook to avoid commiting Jupyter notebook outputs
├── code                   | Python source code for the experiments realized
│   ├── corpus             | Corpora used
│   ├── *.py               | Proposed implemented methods
│   ├── experiments.ipynb  | Experiments source code
│   └── requirement.txt    | Requirement files for external Python packages
└── report                 | Thesis report latex source code
```

## Code

### Requirements

- Python >= 3.8.8 : Programming language
- Pip >= 21.1.3 : Python packet manager
- *the packages in requirements.txt* : See Installation

### Installation

```
cd code
pip install -r requirements.txt
```

### Usage

The experiments (runnable scripts) were written in a Jupyter notebook file format.

Jupyter is installed with the requirements, if you do not already have it.
After installing it, you can simply run, for example, JupyterLab, a web IDE for Jupyter notebooks, with the command below.

```
cd code
jupyter lab
```

## Repository information

This repository history may be not accurate.
This is due to a history rewrite that happened on July the 12th.
The repository was private until this date.

During the git history rewrite, copyrighted files were removed from the tree (copyrighted papers .pdf files), to be allowed on GitHub.
`git filter-branch` was used to the rewrite the history.
