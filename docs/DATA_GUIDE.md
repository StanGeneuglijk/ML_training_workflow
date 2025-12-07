# Data Module Guide

**Delta Lake data generation and loading**

---

## Overview

The `data/` module provides data generation and loading using Delta Lake for ACID transactions, time travel, and schema evolution.

### Module Structure

```
data/
├── delta_lake.py           # Delta Lake operations
└── generate_delta_lake.py  # Data generation
```

---

## Data Generation

### Quick Start

```bash
poetry run python data/generate_delta_lake.py
```

**Creates:** `data/delta_lake/classification_data/` with synthetic classification data (1000 samples, 20 features, binary classification).

### Schema

| Column | Type | Description |
|--------|------|-------------|
| `sample_index` | int64 | Sample identifier |
| `feature_0` to `feature_19` | float64 | Feature values |
| `target` | int64 | Class label (0 or 1) |
| `ingested_at` | timestamp | Ingestion timestamp |
| `ingested_date` | date | Ingestion date |

---

## Data Loading

### Basic Usage

```python
from data import load_data

X, y = load_data("classification_data")
# X.shape: (1000, 20)
# y.shape: (1000,)
```

### Time Travel

```python
# Load specific version
X, y = load_data("classification_data", version=0)
```

---

## Core Functions

### Available Functions

```python
from data import (
    load_data,              
    get_delta_path,         
    read_delta_table,       
    write_to_delta_lake,    
    get_table_info,         
    get_spark_session       
)
```

---

## Integration

### With Training

```python
from data import load_data
from src.orchestrator import run_ml_workflow

X, y = load_data("classification_data")

results = run_ml_workflow(
    feature_specs=feature_specs,
    model_spec=model_spec,
    X=X, y=y
)
```

### With Feature Store

Feature store reads directly from Delta Lake tables via `get_delta_path()`.

---

## Troubleshooting

**Delta Lake table not found:**
```bash
poetry run python data/generate_delta_lake.py
```

---

**Ready for reproducible ML training!** 🎯
