# Feature Store Guide

**Technical guide for Feast feature store integration (Advanced/Optional)**

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [When to Use Feature Store](#when-to-use-feature-store)
4. [Setup](#setup)
5. [Integration](#integration)
6. [Troubleshooting](#troubleshooting)

---

## Overview

The feature store provides **centralized feature management** using Feast, an open-source feature store framework. This is an **optional advanced feature** - the core ML workflow uses direct SQLite loading.

### What is a Feature Store?

A feature store centralizes feature definitions and retrieval:

```
Without Feature Store:        With Feature Store:
────────────────────         ────────────────────
Model A → Load from DB       Model A ─┐
Model B → Load from DB       Model B ─┼→ Feature Store → Parquet
Model C → Load from DB       Model C ─┘
```

### Key Benefits

- ✅ **Centralized features** - Single source of truth for feature definitions
- ✅ **Feature reusability** - Share features across multiple models
- ✅ **Point-in-time correctness** - Historical feature values at specific timestamps
- ✅ **Feature versioning** - Track feature schema changes over time
- ✅ **Governance** - Control feature access and lineage

### Current Status

**⚠️ Feature Store: Optional/Disabled by Default**

The feature store integration exists but is currently:
- **Disabled** in `application_training.py` (entity join issues being resolved)
- **Optional** - Core workflow uses direct SQLite loading
- **Advanced** - For teams needing centralized feature management

**For most users:** Use the standard SQLite data loading (see `DATA_GUIDE.md`)

---

## Architecture

### Data Flow

```
┌─────────────────────────────────────────────────────────┐
│  Step 1: Generate Data                                  │
│  python data/generate_database.py                       │
│  → Creates SQLite: data/database/sample_data.db         │
└────────────────┬────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────┐
│  Step 2A: Direct Loading (Default)                      │
│  from data import load_data                             │
│  X, y = load_data()                                     │
│  → Fast, simple, recommended for single models          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Step 2B: Feature Store (Advanced)                      │
│  from data import export_to_parquet                     │
│  parquet_path = export_to_parquet()                     │
│  → Creates Parquet: data/classification_data.parquet    │
│                                                          │
│  from feature_store import create_feast_manager         │
│  feast_manager = create_feast_manager(...)              │
│  X, y = feast_manager.get_training_data()               │
│  → Centralized, versioned, shareable across models      │
└─────────────────────────────────────────────────────────┘
```

### Module Structure

```
feature_store/
├── __init__.py              # Public API (13 lines)
├── data_sources.py          # Parquet source config (66 lines)
├── feature_definitions.py   # Feature schemas (81 lines)
└── feast_manager.py         # Main operations (246 lines)

Total: ~400 lines (streamlined, focused)
```

### Integration with Data Module

```python
# Feature store uses data module for export
from data import export_to_parquet, get_database_path

# Export SQLite → Parquet
parquet_path = export_to_parquet("classification_data")
# Creates: data/classification_data.parquet

# Feast reads the Parquet file
```

---

## When to Use Feature Store

### ✅ Use Feature Store When:

1. **Multiple Models** - Sharing features across many models
2. **Team Collaboration** - Multiple data scientists need same features
3. **Feature Reuse** - Want to avoid duplicating feature logic
4. **Point-in-Time** - Need historical feature values at specific times
5. **Governance** - Need to track feature lineage and access control
6. **Production Scale** - Many models in production sharing features

### ❌ Skip Feature Store When:

1. **Single Model** - Just training one model
2. **Prototyping** - Rapid iteration, changing features frequently
3. **Small Team** - Solo work or small team with good communication
4. **Simple Features** - Basic features that don't need sharing
5. **Limited Data** - Small datasets (<100K samples)

### Current Project Recommendation

**Use direct loading:**
```python
from data import load_data
X, y = load_data("classification_data")
```

**Why:**
- ✅ Simpler - No feature store setup
- ✅ Faster - Direct SQLite query
- ✅ Sufficient - For demo/single model workflows
- ✅ Reliable - No entity join complexities

**Consider feature store later when:**
- Scaling to multiple models
- Building production ML platform
- Need feature governance

---

## Setup

### Prerequisites

```bash
# Feast is included in dependencies
poetry install
```

### Step 1: Generate and Export Data

```bash
# Generate SQLite database
python data/generate_database.py

# Export to Parquet (for Feast)
python -c "from data import export_to_parquet; export_to_parquet('classification_data')"
```

**Result:**
- `data/database/sample_data.db` - SQLite database
- `data/classification_data.parquet` - Parquet export for Feast

### Step 2: Initialize Feature Store

```python
from feature_store import create_feast_manager

# Create feature store
feast_manager = create_feast_manager(
    repo_path="feature_repo",
    n_features=20,
    initialize=True
)
```

**What this creates:**
```
feature_repo/
├── feature_store.yaml    # Feast configuration
└── data/
    └── registry.db       # Feature registry (SQLite)
```

### Step 3: Verify Setup

```python
# Get training data
X, y = feast_manager.get_training_data()

print(f"Loaded: {X.shape[0]} samples, {X.shape[1]} features")
# Output: Loaded: 1000 samples, 20 features
```

---

## Integration

### Feature Store Components

#### 1. Data Sources (`data_sources.py`)

**Purpose:** Point Feast to Parquet data files

```python
from feature_store.data_sources import (
    get_data_path,
    create_classification_data_source
)

# Get path to Parquet file
parquet_path = get_data_path("classification_data")
# Returns: /path/to/data/classification_data.parquet

# Create Feast FileSource
source = create_classification_data_source()
# Points Feast to the Parquet file
```

#### 2. Feature Definitions (`feature_definitions.py`)

**Purpose:** Define feature schemas and entities

```python
from feature_store.feature_definitions import (
    create_sample_entity,
    create_classification_features
)

# Create entity (join key)
entity = create_sample_entity()
# Entity: sample
# Join key: sample_index

# Create feature view (schema)
feature_view = create_classification_features(n_features=20)
# Features: feature_0 through feature_19 + target
```

**Feature Schema:**
```
Entity: sample (join_key: sample_index)
Features:
  - feature_0: Float64
  - feature_1: Float64
  - ...
  - feature_19: Float64
  - target: Int64
```

#### 3. Feast Manager (`feast_manager.py`)

**Purpose:** Main interface for feature operations

```python
from feature_store import FeastManager

# Initialize
manager = FeastManager(
    repo_path="feature_repo",
    n_features=20
)

# Initialize Feast registry
manager.initialize()

# Get training data
X, y = manager.get_training_data(
    sample_indices=None,  # All samples
    timestamp=None        # Current time
)

# Cleanup
manager.cleanup()
```

### Usage in Training Application

**Current approach (Recommended):**
```python
# application_training.py

from data import load_data

# Direct loading (simple, fast)
X, y = load_data("classification_data")

# Run ML workflow
results = run_ml_workflow(
    feature_specs=feature_specs,
    model_spec=model_spec,
    X=X,
    y=y,
    feature_store_spec=None  # Disabled
)
```

**With feature store (Advanced):**
```python
# application_training.py

from specs import FeatureStoreSpecBuilder

# Configure feature store
fs_spec = FeatureStoreSpecBuilder()\
    .enable()\
    .set_repo_path("feature_repo")\
    .set_n_features(20)\
    .set_initialize_on_start(True)\
    .build()

# Run ML workflow with feature store
results = run_ml_workflow(
    feature_specs=feature_specs,
    model_spec=model_spec,
    X=None,  # Loaded by feature store
    y=None,
    feature_store_spec=fs_spec  # Enabled
)
```

### Public API

```python
from feature_store import (
    # Main class
    FeastManager,
    
    # Factory function
    create_feast_manager,
    
    # Feature definitions
    create_classification_features,
    create_sample_entity,
    
    # Data sources
    create_classification_data_source,
    get_data_path
)
```

---

## Advanced Topics

### Point-in-Time Correctness

**Feature:** Retrieve features as they existed at a specific timestamp

```python
from datetime import datetime

# Get features as of specific date
timestamp = datetime(2025, 11, 1)
X, y = feast_manager.get_training_data(timestamp=timestamp)
```

**Use case:** Training on historical data without data leakage

### Sample Filtering

**Feature:** Retrieve specific samples only

```python
# Get specific samples
sample_indices = [0, 10, 20, 30, 40]
X, y = feast_manager.get_training_data(sample_indices=sample_indices)
# Returns: 5 samples instead of 1000
```

**Use case:** Training on subset of data, stratified sampling

### Feature Store Specification

```python
from specs import FeatureStoreSpecBuilder

# Full configuration
fs_spec = FeatureStoreSpecBuilder()\
    .enable()\
    .set_repo_path("feature_repo")\
    .set_n_features(20)\
    .set_initialize_on_start(True)\
    .set_force_recreate(False)\
    .set_sample_indices([0, 10, 20, 30])\
    .set_timestamp(datetime(2025, 11, 1))\
    .build()
```

---

## Comparison: Direct vs Feature Store

| Aspect | Direct SQLite | Feature Store |
|--------|---------------|---------------|
| **Setup** | ✅ None | ⚠️ Initialize Feast repo |
| **Speed** | ✅ Fast (<10ms) | ⚠️ Slower (~100ms) |
| **Complexity** | ✅ Simple | ⚠️ More complex |
| **Reusability** | ❌ Per-project | ✅ Cross-project |
| **Versioning** | ❌ Manual | ✅ Built-in |
| **Point-in-Time** | ❌ No | ✅ Yes |
| **Governance** | ❌ No | ✅ Yes |
| **Best For** | Single model, prototyping | Multiple models, production |

---

## Troubleshooting

### Issue 1: Database Not Found

```
ERROR: Database not found at .../sample_data.db
```

**Solution:**
```bash
python data/generate_database.py
```

### Issue 2: Parquet File Not Found

```
ERROR: No such file: data/classification_data.parquet
```

**Solution:**
```python
from data import export_to_parquet
export_to_parquet("classification_data")
```

### Issue 3: Entity Join Issues

```
ERROR: No matching entities found
```

**Why this happens:**
- Feast entity join key (`sample_index`) may not match Parquet data
- Timestamp filtering too restrictive

**Solution:**
```python
# Check Parquet file has sample_index column
import pandas as pd
df = pd.read_parquet("data/classification_data.parquet")
print(df.columns)  # Should include 'sample_index'
print(df['sample_index'].head())  # Check values
```

### Issue 4: Feature Store Disabled

```
INFO: Feature store: Disabled (entity join issues to resolve)
```

**Current status:**
- Feature store integration exists but is disabled by default
- Use direct SQLite loading instead
- This is expected behavior for current codebase

**Workaround:**
```python
# Use direct loading
from data import load_data
X, y = load_data("classification_data")
```

### Issue 5: Feast Registry Errors

```
ERROR: Feast registry not found
```

**Solution:**
```bash
# Clean and reinitialize
rm -rf feature_repo/
python -c "
from feature_store import create_feast_manager
manager = create_feast_manager(repo_path='feature_repo', initialize=True)
"
```

---

## Best Practices

### 1. Start Simple

**Recommendation:** Use direct SQLite loading first

```python
from data import load_data
X, y = load_data()  # ✅ Simple, fast, reliable
```

**Add feature store later when needed**

### 2. Export Parquet When Needed

```python
# Only export when using feature store
if use_feature_store:
    from data import export_to_parquet
    export_to_parquet("classification_data")
```

### 3. Clean Up Feast Repos

```bash
# Remove old feature repos between runs
rm -rf feature_repo/
```

### 4. Document Feature Definitions

```python
# Add descriptions to feature views
feature_view = FeatureView(
    name="classification_features",
    description="20 synthetic features for binary classification",
    ...
)
```

### 5. Version Control

**Include in git:**
- `feature_store/*.py` (feature definitions)
- `specs/feature_store_spec.py` (configuration)

**Exclude from git:**
- `feature_repo/` (generated)
- `data/*.parquet` (generated)

---

## Migration Path

### Current: SQLite Only

```python
# Simple, direct loading
from data import load_data
X, y = load_data("classification_data")
```

### Future: Add Feature Store

**When you need:**
- Multiple models sharing features
- Feature governance
- Point-in-time correctness

**Steps:**
1. Export SQLite to Parquet
2. Initialize Feast repository
3. Define feature views
4. Update orchestrator to use feature store
5. Resolve entity join issues
6. Enable in application_training.py

---

## Code Examples

### Example 1: Basic Feature Store Usage

```python
from feature_store import create_feast_manager
from data import export_to_parquet

# 1. Export data to Parquet
export_to_parquet("classification_data")

# 2. Create feature store
manager = create_feast_manager(
    repo_path="feature_repo",
    n_features=20,
    initialize=True
)

# 3. Get training data
X, y = manager.get_training_data()
print(f"Shape: {X.shape}")  # (1000, 20)

# 4. Cleanup
manager.cleanup()
```

### Example 2: With Orchestrator

```python
from specs import FeatureStoreSpecBuilder
from src.orchestrator import run_ml_workflow

# Configure feature store
fs_spec = FeatureStoreSpecBuilder()\
    .enable()\
    .set_repo_path("feature_repo")\
    .set_n_features(20)\
    .build()

# Run workflow (feature store loads data)
results = run_ml_workflow(
    feature_specs=feature_specs,
    model_spec=model_spec,
    X=None,  # Feature store loads
    y=None,
    feature_store_spec=fs_spec
)
```

### Example 3: Point-in-Time Retrieval

```python
from datetime import datetime

# Get historical features
timestamp = datetime(2025, 10, 1)
X, y = manager.get_training_data(timestamp=timestamp)

# Use for backtesting models
```

---

## Summary

### Feature Store Status

**Current:** Optional/Disabled (entity join issues)

**Recommendation:** Use direct SQLite loading (see `DATA_GUIDE.md`)

**Future:** Enable when needed for multi-model workflows

### Integration Points

```
SQLite → export_to_parquet() → Parquet → Feast → Training Data
         ↑                                 ↑
         data module                       feature_store module
```

### When to Revisit

Consider enabling feature store when:
1. ✅ Building multiple models
2. ✅ Need feature governance
3. ✅ Team collaboration on features
4. ✅ Production ML platform

### Current Workflow (Recommended)

```python
# 1. Generate data
$ python data/generate_database.py

# 2. Load directly
from data import load_data
X, y = load_data()

# 3. Train
$ python application_training.py
```

**Simple, fast, reliable!** 🎯

---

## Additional Resources

- **[DATA_GUIDE.md](DATA_GUIDE.md)** - SQLite data architecture (current approach)
- **[MODULE_GUIDE.md](MODULE_GUIDE.md)** - ML modules
- **[ORCHESTRATOR_GUIDE.md](ORCHESTRATOR_GUIDE.md)** - Workflow coordination
- **[Feast Documentation](https://docs.feast.dev/)** - Official Feast docs

---

**Note:** This guide describes the feature store architecture and integration points. For production use, the entity join issues need to be resolved. Until then, use the standard SQLite data loading which is simpler and fully functional.
