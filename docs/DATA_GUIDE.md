# Data Module Technical Guide

**Technical step-by-step breakdown of data generation and loading**

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Data Generation](#data-generation)
4. [Data Loading](#data-loading)
5. [Integration](#integration)

---

## Overview

The `data/` module provides simple, lightweight data generation and loading for the ML workflow. It focuses on **simplicity over complexity** - using SQLite for storage and pandas for loading.

### Module Structure

```
data/
├── __init__.py              # Data loading API (160 lines)
├── generate_database.py     # Synthetic data generation (189 lines)
└── database/               # Generated SQLite database (gitignored)
    └── sample_data.db      # SQLite database file
```

**Design Philosophy:** Keep it simple - SQLite for demo, easy to swap for production

---

## Architecture

### Data Flow

```
1. GENERATE
   generate_database.py
   ↓
   Creates SQLite DB with synthetic classification data
   ↓
2. LOAD
   load_data()
   ↓
   Reads from SQLite → Returns numpy arrays (X, y)
   ↓
3. TRAIN
   ML Workflow
```

### Why SQLite?

**Advantages:**
- ✅ **No infrastructure** - Single file database
- ✅ **Cross-platform** - Works everywhere
- ✅ **SQL queries** - Structured data access
- ✅ **ACID properties** - Data integrity
- ✅ **Fast for demo** - Good for 1K-100K samples
- ✅ **Easy to inspect** - Use sqlite3 CLI or DB Browser

**When to upgrade:**
- Large datasets (>1M samples) → PostgreSQL
- Distributed workloads → Spark + Delta Lake
- Real-time features → Feature store with online serving

---

## Data Generation

**File:** `data/generate_database.py` (189 lines)

**Purpose:** Create SQLite database with synthetic classification data

### Step-by-Step: Database Generation

#### Step 1: Schema Creation

**Tables:**
```sql
-- Dataset metadata
CREATE TABLE datasets (
    dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_name TEXT UNIQUE NOT NULL,
    n_samples INTEGER NOT NULL,
    n_features INTEGER NOT NULL,
    n_classes INTEGER NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Feature data
CREATE TABLE features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL,
    sample_index INTEGER NOT NULL,
    feature_0 REAL,
    feature_1 REAL,
    ... (20 features total)
    feature_19 REAL,
    target INTEGER NOT NULL,
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
);
```

**Design decisions:**
- `dataset_id`: Auto-increment primary key
- `sample_index`: Index of each sample (0, 1, 2, ...)
- Fixed schema: 20 features (feature_0 through feature_19)
- `target`: Integer class label (0 or 1 for binary)
- Foreign key constraint for referential integrity

#### Step 2: Synthetic Data Generation

**Function:** `populate_dataset()`

```python
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,      # 50% informative features
    n_redundant=5,          # 25% redundant features
    n_classes=2,            # Binary classification
    random_state=42         # Reproducible
)
```

**What `make_classification` creates:**
- 1000 samples
- 20 features (continuous, normalized)
- Binary target (0 or 1)
- Realistic class imbalance and feature correlations
- Controlled difficulty (n_informative determines separability)

#### Step 3: Database Population

**Insert dataset metadata:**
```python
cursor.execute("""
    INSERT INTO datasets (dataset_name, n_samples, n_features, n_classes, description)
    VALUES (?, ?, ?, ?, ?)
""", ("classification_data", 1000, 20, 2, "Synthetic classification dataset"))

dataset_id = cursor.lastrowid
```

**Insert feature data:**
```python
# Batch insert all samples
rows = [(idx, *features, int(y[idx])) for idx, features in enumerate(X)]

cursor.executemany("""
    INSERT INTO features (
        dataset_id, sample_index,
        feature_0, feature_1, ..., feature_19,
        target
    )
    VALUES (?, ?, ?, ?, ..., ?, ?)
""", [(dataset_id, *row) for row in rows])

conn.commit()
```

**Performance:**
- `executemany()` for batch inserts
- Single transaction for all 1000 samples
- ~0.1 seconds to insert

#### Step 4: Run Generation

```bash
python data/generate_database.py
```

**Output:**
```
Sample database created.
Location: /Users/.../data/database/sample_data.db
```

**Result:** SQLite database ready for loading

---

## Data Loading

**File:** `data/__init__.py` (160 lines)

**Purpose:** Load data from SQLite database as numpy arrays

### Public API

```python
from data import load_data, get_database_path, export_to_parquet

# Load training data
X, y = load_data(dataset_name="classification_data")

# Get database path
db_path = get_database_path()

# Export to Parquet (for Feast)
parquet_path = export_to_parquet(dataset_name="classification_data")
```

### Step-by-Step: Data Loading

#### Function: load_data()

**Step 1: Get Database Path**
```python
def get_database_path() -> Path:
    """Get path to SQLite database."""
    return Path(__file__).parent / "database" / "sample_data.db"

db_path = get_database_path()
```

**Step 2: Check Database Exists**
```python
if not db_path.exists():
    raise FileNotFoundError(
        f"Database not found at {db_path}. "
        "Run 'python data/generate_database.py' first."
    )
```

**Step 3: Connect and Query Dataset ID**
```python
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get dataset_id
cursor.execute(
    "SELECT dataset_id FROM datasets WHERE dataset_name = ?",
    (dataset_name,)
)
result = cursor.fetchone()

if result is None:
    conn.close()
    raise ValueError(f"Dataset '{dataset_name}' not found")

dataset_id = result[0]
```

**Step 4: Load Features**
```python
# Query all features for this dataset
query = """
    SELECT 
        feature_0, feature_1, feature_2, ..., feature_19,
        target
    FROM features
    WHERE dataset_id = ?
    ORDER BY sample_index
"""

df = pd.read_sql_query(query, conn, params=(dataset_id,))
conn.close()
```

**What this returns:**
```python
# DataFrame with shape (1000, 21)
   feature_0  feature_1  ...  feature_19  target
0   1.470848  -0.360450  ...   -0.833142       1
1  -0.977278   0.950088  ...    0.123456       0
...
```

**Step 5: Split into X and y**
```python
feature_cols = [f'feature_{i}' for i in range(20)]
X = df[feature_cols].values  # Shape: (1000, 20)
y = df['target'].values       # Shape: (1000,)

return X, y
```

**Final output:**
- `X`: numpy array, shape (1000, 20), dtype float64
- `y`: numpy array, shape (1000,), dtype int64

---

## Integration

### With Training Application

```python
# In application_training.py
from data import load_data

# Load data
X, y = load_data(dataset_name="classification_data")
# X.shape: (1000, 20)
# y.shape: (1000,)

# Ready for ML workflow
results = run_ml_workflow(
    feature_specs=feature_specs,
    model_spec=model_spec,
    X=X,
    y=y,
    ...
)
```

### With Feature Store (Future)

```python
# Export SQLite data to Parquet for Feast
from data import export_to_parquet

parquet_path = export_to_parquet(dataset_name="classification_data")
# Creates: data/classification_data.parquet

# Feast can read this
# feature_source = FileSource(path=str(parquet_path), ...)
```

### With Tests

```python
# Easy to create test databases
import sqlite3
from pathlib import Path

def create_test_database(tmp_path: Path) -> Path:
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    
    # Create schema
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE datasets (...)")
    cursor.execute("CREATE TABLE features (...)")
    
    # Insert test data
    cursor.execute("INSERT INTO datasets (...) VALUES (...)")
    # ...
    
    conn.commit()
    conn.close()
    return db_path
```

---

## Database Schema Details

### datasets Table

| Column | Type | Constraints | Purpose |
|--------|------|-------------|---------|
| `dataset_id` | INTEGER | PRIMARY KEY, AUTOINCREMENT | Unique dataset identifier |
| `dataset_name` | TEXT | UNIQUE, NOT NULL | Human-readable name |
| `n_samples` | INTEGER | NOT NULL | Number of samples |
| `n_features` | INTEGER | NOT NULL | Number of features (20) |
| `n_classes` | INTEGER | NOT NULL | Number of classes (2) |
| `description` | TEXT | NULL | Optional description |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Creation time |

**Example row:**
```
dataset_id: 1
dataset_name: "classification_data"
n_samples: 1000
n_features: 20
n_classes: 2
description: "Synthetic classification dataset"
created_at: "2025-11-08 12:27:46"
```

### features Table

| Column | Type | Constraints | Purpose |
|--------|------|-------------|---------|
| `id` | INTEGER | PRIMARY KEY, AUTOINCREMENT | Row identifier |
| `dataset_id` | INTEGER | FOREIGN KEY, NOT NULL | References datasets |
| `sample_index` | INTEGER | NOT NULL | Sample index (0-999) |
| `feature_0` | REAL | NULL | First feature value |
| ... | REAL | NULL | ... |
| `feature_19` | REAL | NULL | Last feature value |
| `target` | INTEGER | NOT NULL | Class label (0 or 1) |

**Example row:**
```
id: 1
dataset_id: 1
sample_index: 0
feature_0: 1.470848
feature_1: -0.360450
...
feature_19: -0.833142
target: 1
```

### Indexes and Performance

**No explicit indexes needed for demo:**
- 1000 samples is small enough
- Full table scan is fast (<10ms)
- Primary keys automatically indexed

**For production (>100K samples):**
```sql
CREATE INDEX idx_dataset_sample ON features(dataset_id, sample_index);
CREATE INDEX idx_dataset ON features(dataset_id);
```

---

## Data Characteristics

### Synthetic Dataset Properties

**Generated by:** `sklearn.datasets.make_classification`

**Properties:**
- **Samples**: 1000
- **Features**: 20 continuous features
- **Distribution**: Approximately Gaussian
- **Correlation**: Features have realistic correlations
- **Classes**: Binary (0, 1)
- **Balance**: Roughly balanced (500/500)
- **Separability**: ~92% achievable accuracy (controllable)

**Feature statistics:**
```python
# Typical ranges after generation:
feature_mean ≈ 0.0
feature_std ≈ 1.0
feature_min ≈ -3.0
feature_max ≈ +3.0
```

### Quality Checks

**After generation, verify:**
```python
import sqlite3
conn = sqlite3.connect("data/database/sample_data.db")

# Check row count
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM features WHERE dataset_id = 1")
print(f"Rows: {cursor.fetchone()[0]}")  # Should be 1000

# Check class distribution
cursor.execute("SELECT target, COUNT(*) FROM features WHERE dataset_id = 1 GROUP BY target")
for row in cursor.fetchall():
    print(f"Class {row[0]}: {row[1]} samples")

conn.close()
```

---

## Advanced Topics

### Custom Datasets

**Modify `generate_database.py` for custom data:**

```python
# Change dataset size
populate_dataset(
    db_path=db_path,
    dataset_name="large_dataset",
    n_samples=10000,  # ← 10x larger
    n_features=20,
    n_classes=2,
    random_state=42
)

# Multi-class problem
populate_dataset(
    db_path=db_path,
    dataset_name="multiclass_data",
    n_samples=1000,
    n_features=20,
    n_classes=5,  # ← 5 classes
    random_state=42
)
```

### Multiple Datasets

```python
# Store multiple datasets in same database
populate_dataset(db_path, "train_data", n_samples=800, ...)
populate_dataset(db_path, "val_data", n_samples=100, ...)
populate_dataset(db_path, "test_data", n_samples=100, ...)

# Load specific dataset
X_train, y_train = load_data("train_data")
X_val, y_val = load_data("val_data")
X_test, y_test = load_data("test_data")
```

### Export to Parquet

**For Feast feature store:**

```python
from data import export_to_parquet

# Export SQLite → Parquet
parquet_path = export_to_parquet(dataset_name="classification_data")
# Creates: data/classification_data.parquet

# Includes:
# - All features (feature_0 through feature_19)
# - sample_index (for joins)
# - target
# - ingested_at (timestamp for Feast)
```

**What happens:**
1. Load data from SQLite
2. Add `ingested_at` timestamp column
3. Write to Parquet format
4. Return path to file

**Use case:** Feast FileSource requires Parquet format

---

## Best Practices

### 1. Regenerate Database for Clean State

```bash
# Remove old database
rm -f data/database/sample_data.db

# Generate fresh data
python data/generate_database.py
```

**When to regenerate:**
- Testing different dataset sizes
- Want different random seed
- Database corrupted
- Schema changed

### 2. Check Data Before Training

```python
from data import load_data

try:
    X, y = load_data()
    print(f"✓ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
except FileNotFoundError:
    print("✗ Database not found - run generate_database.py")
except ValueError as e:
    print(f"✗ Data error: {e}")
```

### 3. Use Context Managers for DB Access

**If extending data module:**
```python
import sqlite3
from contextlib import closing

with sqlite3.connect(db_path) as conn:
    with closing(conn.cursor()) as cursor:
        cursor.execute("SELECT * FROM datasets")
        # Connection automatically closed
```

### 4. Git Ignore Generated Data

**Already in `.gitignore`:**
```gitignore
data/*.db
data/*.csv
data/*.parquet
data/database/
```

**Why:** Generated data shouldn't be in version control

---

## Migration to Production Data

### From SQLite to PostgreSQL

**Step 1: Update schema**
```sql
-- PostgreSQL equivalent
CREATE TABLE datasets (
    dataset_id SERIAL PRIMARY KEY,
    dataset_name VARCHAR(255) UNIQUE NOT NULL,
    ...
);
```

**Step 2: Update load_data()**
```python
# In data/__init__.py
import psycopg2

def load_data(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    conn = psycopg2.connect(
        host="postgres.example.com",
        database="ml_data",
        user="ml_user",
        password=os.getenv("DB_PASSWORD")
    )
    
    # Same SQL query works!
    query = "SELECT feature_0, ... FROM features WHERE ..."
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df[feature_cols].values, df['target'].values
```

**No changes needed in:**
- `application_training.py`
- `src/orchestrator.py`
- Any ML modules

**Why:** Clean data interface!

### From Synthetic to Real Data

**Step 1: Load real data**
```python
# In generate_database.py
import pandas as pd

# Load from CSV
real_data = pd.read_csv("production_data.csv")

# Prepare features and target
X = real_data.drop('target', axis=1).values
y = real_data['target'].values

# Insert into database (same code as synthetic)
for idx, (features, target) in enumerate(zip(X, y)):
    cursor.execute(...)
```

**Step 2: Validate schema matches**
```python
assert X.shape[1] == 20, "Must have 20 features"
assert y.dtype in [np.int32, np.int64], "Target must be integer"
```

**Step 3: Test**
```python
X_loaded, y_loaded = load_data("production_data")
assert X_loaded.shape == X.shape
assert (y_loaded == y).all()
```

---

## Comparison: SQLite vs Alternatives

| Feature | SQLite (Current) | Spark + Delta | PostgreSQL | Feature Store |
|---------|------------------|---------------|------------|---------------|
| **Setup** | ✅ None | ❌ Complex | ⚠️ Medium | ⚠️ Medium |
| **Performance** | ✅ <1M samples | ✅ Unlimited | ✅ Millions | ✅ Real-time |
| **Versioning** | ❌ Manual | ✅ Built-in | ❌ Manual | ✅ Built-in |
| **ACID** | ✅ Yes | ✅ Yes | ✅ Yes | ⚠️ Eventually |
| **Schema Evolution** | ⚠️ Manual | ✅ Automatic | ⚠️ Migrations | ✅ Automatic |
| **Best For** | Demo, <100K | Big Data | Production | Feature Platform |

**Current choice:** SQLite (perfect for ML engineering demo)

**Upgrade path:** Same interface, swap implementation

---

## Troubleshooting

### Database Not Found

```
ERROR: Database not found at .../sample_data.db
```

**Solution:**
```bash
python data/generate_database.py
```

### Dataset Not Found

```
ERROR: Dataset 'classification_data' not found in database
```

**Check available datasets:**
```bash
sqlite3 data/database/sample_data.db "SELECT dataset_name FROM datasets;"
```

### Schema Mismatch

```
ERROR: no such column: feature_20
```

**Solution:** Ensure `n_features=20` matches schema (feature_0 through feature_19)

### Empty Database

```bash
# Check if database has data
sqlite3 data/database/sample_data.db "SELECT COUNT(*) FROM features;"
```

**If 0:** Regenerate database

---

## Summary

### Data Module Responsibilities

✅ **Generate** - Create synthetic classification data
✅ **Store** - Persist in SQLite database
✅ **Load** - Read as numpy arrays for ML
✅ **Export** - Convert to Parquet for feature store

### Code Size

- `generate_database.py`: 189 lines (data generation)
- `__init__.py`: 160 lines (data loading)
- **Total: 349 lines** (simple and focused!)

### Design Highlights

✅ **Simple** - SQLite, no infrastructure
✅ **Fast** - Loads 1000 samples in <10ms
✅ **Reproducible** - Fixed random seed
✅ **Testable** - Easy to create test databases
✅ **Upgradeable** - Clean interface for production swap

**Perfect for:** ML engineering focus without data engineering complexity! 🎯

---

## Next Steps

- **Module Guide**: See `MODULE_GUIDE.md` for ML modules
- **Specs Guide**: See `SPECS_GUIDE.md` for configuration
- **Orchestrator**: See `ORCHESTRATOR_GUIDE.md` for workflow coordination
- **Feature Store**: See `FEATURE_STORE.md` for Feast integration (future)
