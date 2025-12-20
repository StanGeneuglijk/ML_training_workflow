# Feature Store Guide

**Feast feature store integration with Delta Lake**

---

## Overview

Centralized feature management using Feast for sharing features across models with version control and governance.

### Architecture

```
Delta Lake → Feast FileSource → FeatureView → Training Data
```

### Module Structure

```
feature_store/
├── data_sources.py          # File source creation
├── feature_definitions.py   # Entity/schema/view creation
└── feast_manager.py         # Main operations
```

---

## When to Use

**✅ Use when:**
- Multiple models sharing features
- Need feature versioning and governance
- Point-in-time correctness required

**❌ Skip when:**
- Single model or prototyping
- Small team with simple features

---

## Setup

### Step 1: Generate Data

```bash
poetry run python data/generate_delta_lake.py
```

### Step 2: Configure

```python
from specs_training import FeatureStoreSpecBuilder

feature_store_spec = (
    FeatureStoreSpecBuilder()
    .enable()
    .set_repo_path("feature_repo")
    .set_dataset_name("classification_data")
    .set_n_features(20)
    .set_initialize_on_start(True)
    .build()
)
```

### Step 3: Use in Training

```python
from src_training.orchestrator import run_ml_workflow

results = run_ml_workflow(
    feature_specs=feature_specs,
    model_spec=model_spec,
    X=None, 
    y=None,
    feature_store_spec=feature_store_spec
)
```

---

## Core API

```python
from feature_store import (
    FeastManager,
    create_feast_manager,
    create_entity,
    create_schema,
    create_feature_view,
)

# Create manager
manager = create_feast_manager(
    repo_path="feature_repo",
    project_name="ml_workflow",
    initialize=True
)

# Get training data
X, y = manager.get_training_data(
    entity_df=entity_df,
    feature_names=[f"feature_{i}" for i in range(20)],
    target_name="target"
)

# Cleanup
manager.cleanup()
```

---

## Key Features

- **Generic design** - Works with any dataset
- **Delta Lake integration** - ACID transactions and time travel
- **Automatic setup** - Orchestrator handles registration
- **Point-in-time** - Historical feature retrieval

---

## Troubleshooting

**Delta Lake table not found:**
```bash
poetry run python data/generate_delta_lake.py
```

**Feast registry errors:**
```bash
rm -rf feature_repo/
```

---

**Ready for production feature management!** 🎯
