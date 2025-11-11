# Applications Guide

ML Workflow Version 1 provides two main applications for the complete ML lifecycle.

---

## 📊 application_training.py

**Purpose:** Train models and register them in MLflow

### Usage

```bash
# Basic training
poetry run python application_training.py

# With feature store
poetry run python application_training.py --feature-store
```

### What it does

1. Loads data from SQLite database (or Feature Store if enabled)
2. Creates feature specifications
3. Builds ML pipeline
4. Trains model with hyperparameter tuning
5. Applies calibration
6. Validates with cross-validation
7. **Registers model in MLflow Model Registry**
8. Logs all metadata to MLflow

### Output

- Trained model registered in MLflow
- Experiment tracking data
- Model metrics and artifacts
- Cross-validation results

### Next Steps After Training

```bash
# View results
poetry run mlflow ui --port 5000

# Then navigate to http://localhost:5000
# Check "Models" tab for registered models
# Promote to "Production" stage when ready
```

---

## 🔮 application_inference_serving.py

**Purpose:** Load trained models and make predictions

### Modes

#### 1. List Models

View all registered models:

```bash
poetry run python application_inference_serving.py list
```

#### 2. Interactive Mode

Test predictions on sample data:

```bash
poetry run python application_inference_serving.py interactive \
    --model-name gradient_boosting_classifier_registered \
    --model-stage Production
```

**Use case:** Quick testing, model validation

#### 3. Batch Mode

Process large datasets:

```bash
poetry run python application_inference_serving.py batch \
    --input data/test.csv \
    --output data/predictions.csv \
    --model-name gradient_boosting_classifier_registered \
    --model-stage Production \
    --return-proba
```

**Use case:** 
- Daily/weekly prediction jobs
- Processing historical data
- Large-scale offline predictions

#### 4. Serve Mode

Start REST API server:

```bash
poetry run python application_inference_serving.py serve \
    --model-name gradient_boosting_classifier_registered \
    --model-stage Production \
    --port 8000
```

**Use case:**
- Real-time predictions
- Web service integration
- Mobile app backend

---

## 🔄 Complete Workflow

### Step 1: Train Model

```bash
poetry run python application_training.py
```

### Step 2: Review & Promote

```bash
# Start MLflow UI
poetry run mlflow ui --port 5000

# In browser (http://localhost:5000):
# 1. Go to "Models" tab
# 2. Find your model
# 3. Click on the version
# 4. Click "Stage" → "Transition to Production"
```

### Step 3: Make Predictions

```bash
# Test predictions
poetry run python application_inference_serving.py interactive \
    --model-name gradient_boosting_classifier_registered \
    --model-stage Production

# Or process files
poetry run python application_inference_serving.py batch \
    --input data/test.csv \
    --output data/predictions.csv \
    --model-name gradient_boosting_classifier_registered \
    --model-stage Production

# Or start API server
poetry run python application_inference_serving.py serve \
    --model-name gradient_boosting_classifier_registered \
    --model-stage Production \
    --port 8000
```

---

## 🎯 Common Options

### Training Options

```bash
# With feature store
poetry run python application_training.py --feature-store

# Short flag
poetry run python application_training.py -fs
```

### Inference Options

```bash
# Specific version
--model-version 3

# Different stage
--model-stage Staging

# Custom MLflow URI
--tracking-uri http://mlflow-server:5000

# For batch mode
--input-format parquet
--output-format parquet
--batch-size 500
--return-proba

# For serve mode
--host 0.0.0.0
--port 8000
--reload  # development mode with auto-reload
```

---

## 📚 Examples

### Training Pipeline

```bash
# 1. First time: generate data
poetry run python data/generate_database.py

# 2. Train model
poetry run python application_training.py

# 3. View results
poetry run mlflow ui --port 5000
```

### Production Inference

```bash
# 1. List available models
poetry run python application_inference_serving.py list

# 2. Start API server
poetry run python application_inference_serving.py serve \
    --model-name my_classifier \
    --model-stage Production \
    --port 8000

# 3. Make predictions (in another terminal)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"feature_0": 0.5, "feature_1": -1.2}, "return_proba": true}'
```

### Batch Processing Pipeline

```bash
# 1. Process today's data
poetry run python application_inference_serving.py batch \
    --input data/daily/2025-11-03.csv \
    --output data/predictions/2025-11-03.csv \
    --model-name my_classifier \
    --model-stage Production \
    --return-proba

# 2. Schedule with cron (add to crontab)
# 0 2 * * * cd /path/to/project && poetry run python application_inference_serving.py batch ...
```

---

## 🔧 Troubleshooting

### Model Not Found

If you see "model not found" errors:

```bash
# 1. List all models
poetry run python application_inference_serving.py list

# 2. Check MLflow UI
poetry run mlflow ui --port 5000

# 3. Make sure model is in the correct stage
# 4. Try with --model-stage Staging if Production is empty
```

### Training Issues

```bash
# Check data exists
poetry run python data/generate_database.py

# View logs
# Check logs/application.log (if logging configured)
```

### API Connection Issues

```bash
# Check port availability
lsof -i :8000

# Try different port
poetry run python application_inference_serving.py serve \
    --model-name my_classifier \
    --port 8001

# Check API docs
# Navigate to http://localhost:8000/docs
```

---

## 📖 Additional Resources

- **[README.md](README.md)** - Project overview
- **[INFERENCE_QUICKSTART.md](INFERENCE_QUICKSTART.md)** - 5-minute quick start
- **[docs/INFERENCE_AND_SERVING.md](docs/INFERENCE_AND_SERVING.md)** - Complete serving guide
- **[docs/MLFLOW.md](docs/MLFLOW.md)** - MLflow integration details
- **[docs/FEATURE_STORE.md](docs/FEATURE_STORE.md)** - Feature store guide

---

## 💡 Tips

1. **Always review results in MLflow UI before promoting to Production**
2. **Use Staging stage for testing inference before Production**
3. **Enable --reload flag during API development**
4. **Use batch mode for large datasets instead of API calls**
5. **Monitor API performance with `/health` endpoint**
6. **Keep track of model versions for reproducibility**
7. **Use feature store for consistent features across training and inference**

