Here's a `README.md` file for your project. This file provides an overview, installation instructions, usage details, and other relevant information.

```markdown
# Loan Repayment Prediction Web Application

## Overview

This project is a web application for predicting loan repayment status using machine learning models. The application allows users to input various loan parameters and get predictions on whether a loan will be repaid or not. It uses models trained with historical loan data and provides evaluation metrics for different models.

## Technologies Used

- **Python**: The primary programming language.
- **FastAPI**: Web framework for building the API.
- **scikit-learn**: Machine learning library for training models.
- **joblib**: For saving and loading models.
- **HTML/CSS**: For the frontend of the web application.

## Features

- **Predict Loan Repayment**: Users can input loan details and receive a prediction on whether the loan will be repaid.
- **Model Evaluation**: Evaluate the performance of different machine learning models on the test dataset.
- **Data Preprocessing**: The application includes data preprocessing steps to prepare data for training and predictions.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Thilakkumar48/loan-repayment-prediction.git
   cd loan-repayment-prediction
   ```

2. **Set Up a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Dataset**

   Ensure you have the `loan_data.csv` file in the project directory. This dataset should contain features such as `fico`, `int_rate`, `installment`, etc.

## Usage

1. **Start the FastAPI Server**

   ```bash
   uvicorn main:app --reload
   ```

2. **Access the Web Application**

   Open your web browser and go to `http://127.0.0.1:8000`.

3. **Using the Application**

   - **Predict Loan Repayment**: Navigate to the form, input loan details, select a model, and submit to get the prediction.
   - **Evaluate Models**: Use the `/evaluate/{model_name}` endpoint to get evaluation metrics for a specific model.

## Endpoints

- **GET /**: Render the main page with the prediction form.
- **POST /predict**: Submit loan details and get a prediction.
- **GET /evaluate/{model_name}**: Evaluate a specific model and get performance metrics.

## Project Structure

- `main.py`: Contains the FastAPI application and model logic.
- `loan_data.csv`: Dataset used for training the models.
- `templates/`: Directory containing HTML templates (`index.html` and `result.html`).
- `models/`: Directory where trained models are saved (`decision_tree.pkl`, `random_forest.pkl`, `gradient_boosting.pkl`).
- `requirements.txt`: Lists the Python dependencies for the project.

## Troubleshooting

- **Feature Mismatch Error**: Ensure that the features provided during prediction match those used for training.
- **Model Not Found**: Check that the selected model name matches one of the available models.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The scikit-learn documentation for machine learning models.
- FastAPI documentation for building the web application.
```

### Notes:
- Replace `https://github.com/Thilakkumar48/loan-repayment-prediction.git` with the actual URL of your GitHub repository if different.
- Ensure `requirements.txt` includes all necessary Python libraries (`fastapi`, `scikit-learn`, `joblib`, `uvicorn`, etc.).
- Adjust the file paths and names if they differ from the ones used in your project.

Feel free to customize the `README.md` file to better suit your project specifics and requirements.
