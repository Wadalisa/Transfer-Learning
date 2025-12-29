# 🧠⚔️ TRANSFER LEARNING — MAIN QUEST

> *“Reuse knowledge. Reduce grind. Optimize evolution.”*

---

## 🗺️ Quest Overview

Training Machine Learning models is a **high-cost grind** — large datasets, long training times, and heavy compute requirements.  
This project applies **Transfer Learning** within a **Genetic Programming (GP)** framework to reduce training effort by reusing knowledge from a **Source Task** and adapting it to a **Target Task**.

A pre-trained GP population is transferred, refined, and evolved to solve a related problem more efficiently.

---

## 📊 DATASETS — QUEST MAP

### 🧠 Source GP Dataset

- **Name:** `227_cpu_small`
- **Observations:** 8,192
- **Variables:** 13
- **Attributes:** Continuous
- **Missing / NaN Values:** Zero values present
- **Duplicate Rows:** None

---

### 🎯 Target GP Dataset

- **Name:** `197_cpu_act`
- **Observations:** 8,192
- **Variables:** 20
- **Attributes:** Continuous
- **Missing / NaN Values:** Zero values present
- **Duplicate Rows:** None

---

### 🛠️ Pre-Processing Buffs

- Duplicate rows **removed** to reduce bias
- **Min–Max Normalization** applied to all features

Normalization prevents features with large numeric ranges (e.g. `freemem`) from overpowering smaller-scale features (e.g. `runqsz`), ensuring fair contribution during evolution.

---

## 🌳 MODEL REPRESENTATION — SKILL TREE

The GP regressor is represented as an **Expression Tree**:

- **Internal Nodes:** Operators  
- **Leaf Nodes:** Operands (terminals)

Each individual consists of:
- A **root node** selected from the functional set
- Child nodes selected from functional or terminal sets

Expression trees offer flexibility and allow easy manipulation during evolution.

### 🌱 Initial Population

- **Generation Method:** Growth Method
- **Initial Tree Depth:** `0`
- Ensures simple expressions at the start of evolution

---

## 📐 FITNESS FUNCTION — DAMAGE CALCULATION

**Mean Absolute Error (MAE)** is used as the fitness function.

**Why MAE?**
- Always **non-negative**
- Robust to outliers
- Ideal for regression-based GP evaluation

Each individual expression is evaluated using MAE — **lower is better**.

---

## 🎯 SELECTION METHOD — PARTY RECRUITMENT

**Tournament Selection** is used:

- Randomly selects `k` individuals
- The individual with the **lowest fitness score** wins
- One parent returned per tournament

**Why Tournament Selection?**
- Simple and efficient
- Computational complexity: **O(n)**
- Maintains balanced selection pressure

---

## 🧬 GENETIC OPERATORS — EVOLUTION MECHANICS

Two operators drive evolution:

- **Crossover** → Exploitation  
- **Mutation** → Exploration  

This balance ensures both refinement of strong individuals and discovery of new solutions.

---

### 🔀 Crossover (Subtree Swap)

- Random subtrees are selected from two parents
- Subtrees are swapped to form offspring
- Controlled by the **Crossover Rate**

---

### 🧫 Mutation (Point Mutation)

- A random node in the tree is replaced
- Introduces novelty and diversity
- Controlled by the **Mutation Rate**

---

## 🔁 TRANSFER LEARNING — KNOWLEDGE PASSIVE

1. Source GP is trained on the source dataset
2. Final source population is extracted
3. A portion is transferred to the Target GP
4. Remaining population is generated via growth method
5. Target GP evolves over `g` generations
6. Final fitness is evaluated on the target dataset

The **Transfer Rate** determines how much knowledge is reused from the source population.

---

## ⏹️ TERMINATION CONDITION

- Evolution ends after a **fixed number of generations**

---

## ⚙️ PARAMETERS — BUILD STATS

### 👥 Population Size

- **Source GP:** `50`
  - Balances diversity and training efficiency

- **Target GP:** `25`
  - Encourages reuse of transferred individuals while allowing novelty

---

### 🔀 Crossover Rate — `60%`

- If random value < 0.6 → crossover occurs
- Otherwise, individual is copied unchanged

---

### 🧫 Mutation Rate — `35%`

- Higher rate to encourage exploration
- Reduces risk of premature convergence
- Point mutation affects only one node

---

### 🏆 Elitism Rate — `5%`

- Best individuals are preserved
- Prevents loss of high-fitness solutions

---

### 🌲 Maximum Tree Depth — `3`

- Prevents excessive tree growth
- Reduces noise and bloat
- Selected after multiple simulations

---

### 🌱 Initial Population Method

- **Growth Method**

---

### 🎯 Tournament Size

- **k = 2**
- Balances selection pressure and diversity

---

### ➕ Functional & Terminal Sets

**Functional Set:**
{ +, -, *, /, sqrt, cos, sin, log }
- Includes both binary and unary operators
- Operators are stored as strings for expression tree construction

**Terminal Set:**
f(xi) for i in range(1, num_features+1) + ["c"]
- Represents all features in the dataset and a constant `c`
- Varies depending on the GP and dataset used:
  - **Source GP:** `x1, x2, ..., x13` + constant `c`
  - **Target GP:** `x1, x2, ..., x20` + constant `c`

**Safety Handling:**
- Division by zero → returns `1`
- Square root of negative values → absolute value applied first

This ensures robust evaluation and prevents crashes during evolution.

---

## 🏁 QUEST STATUS

🧩 **Main Quest:** Transfer Learning with Genetic Programming  
🚀 **Objective:** Reduce training cost while maintaining performance  
🏆 **Reward:** Efficient knowledge reuse across related tasks

---

*Honours-level project — built for evolution, not brute force.*
