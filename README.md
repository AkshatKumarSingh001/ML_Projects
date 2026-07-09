# Student Performance Analysis

An end-to-end machine learning project that predicts a student's math score from a small profile: gender, ethnicity group, parental education, lunch type, test prep course, reading score, and writing score.

The project includes:

- Data ingestion and preprocessing pipelines
- Model training and evaluation
- Serialized preprocessing and model artifacts
- A Flask web application for prediction
- A polished frontend for interactive input and result display

## Project Overview

This repository demonstrates a complete ML workflow:

1. Raw student performance data is prepared and cleaned.
2. A preprocessing pipeline transforms categorical and numeric features.
3. Multiple regression models are evaluated.
4. The best model and preprocessor are saved to the `artifacts/` folder.
5. A Flask application loads those artifacts and serves predictions through a browser UI.

## Features

- Predicts math score from seven input features
- Uses a trained preprocessing pipeline and regression model
- Simple browser-based interface built with Flask and HTML/CSS/JavaScript
- Persistent training artifacts for inference reuse
- Modular source code organized by responsibility

## Repository Structure

```text
.
|-- app.py
|-- README.md
|-- requirements.txt
|-- setup.py
|-- artifacts/
|-- data/
|-- logs/
|-- notebook/
|-- src/
|   |-- exception.py
|   |-- logger.py
|   |-- utils.py
|   |-- components/
|   |   |-- data_ingestion.py
|   |   |-- data_transformation.py
|   |   `-- model_trainer.py
|   `-- pipeline/
|       |-- predict_pipeline.py
|       `-- train_pipeline.py
`-- templates/
	`-- home.html
```

## Requirements

- Python 3.10+ recommended
- A virtual environment is strongly recommended
- Packages listed in `requirements.txt`

## Installation

### 1. Clone or open the project

Open the project folder in VS Code or clone it locally.

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

If you prefer editable installation for the local package:

```powershell
pip install -e .
```

## Running the Application

Start the Flask app from the project root:

```powershell
.venv\Scripts\python.exe app.py
```

Then open the prediction page in your browser:

```text
http://127.0.0.1:5000/predictdata
```

## How It Works

The web form sends the following values to the backend:

- `gender`
- `race_ethnicity`
- `parental_level_of_education`
- `lunch`
- `test_preparation_course`
- `reading_score`
- `writing_score`

The backend converts the input into a pandas DataFrame, applies the saved preprocessor, and uses the saved model to generate a math score prediction.

## Model Artifacts

The training pipeline stores reusable objects in `artifacts/`:

- `model.pkl` - trained regression model
- `preprocessor.pkl` - preprocessing pipeline
- `train.csv` and `test.csv` - processed train/test data splits

These artifacts allow the Flask app to run inference without retraining.

## Development Notes

- The project uses a custom exception class and logging helper under `src/`
- Training and inference are separated into different pipeline modules
- The frontend is rendered with Flask templates and standard browser JavaScript

## Troubleshooting

- If the app does not start, confirm that the virtual environment is activated and dependencies are installed.
- If prediction fails, make sure the `artifacts/` folder contains the trained model and preprocessor files.
- If you see schema-related errors, ensure the frontend field names match the backend feature names.

## License

This project is licensed under the [MIT License](LICENSE).

## Author

Akshat Kumar Singh
