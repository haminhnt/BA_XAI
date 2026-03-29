# AI-Based KPI Reporting with Explainable AI

Bachelor's Thesis Project

## About This Project

This repository contains the code for my Bachelor's thesis:

**"Enhancing Trust and Adoption in AI-Based KPI Reporting through Explainable AI (XAI)"**

The project demonstrates:
- Predictive analytics using a Random Forest model for weekly supermarket sales forecasting
- Integration of Explainable AI (XAI) techniques to make AI predictions transparent
- An interactive executive dashboard with KPI reporting, sales driver analysis, and AI-generated natural language explanations

## Main Features

- Random Forest regression model trained on Walmart sales data
- SHAP-based waterfall explanations for sales drivers
- Interactive KPI summary cards (Actual vs Predicted, Feature comparison)
- AI-generated manager-friendly narrative explanations (powered by Ollama + Llama 3.2)
- Executive-style dashboard built with ExplainerDashboard + Dash
  
## Technologies Used

- Python 3
- scikit-learn (Random Forest)
- explainerdashboard
- Dash + Plotly
- Ollama + Llama 3.2 (for natural language explanations)

## Repository Contents

- `app.py` → Main dashboard application (Streamlit-style layout using Dash + ExplainerDashboard)
- `Walmart_Sales.csv` → Dataset used for training and demonstration
- `requirements.txt` → List of required Python packages

## How to Run

### 1. Clone the repository:
   ``` bash
   git clone https://github.com/haminhnt/BA_XAI.git
   cd BA_XAI
   ```
2. Install Python Dependencies:
   ```bash
   pip install -r requirements.txt
   ```` 
4. Install and run Ollama (local LLMs API):
   Download and install Ollama from https://ollama.com
   Pull the required model:
   ```bash
    ollama pull llama3.2
   ```
   Start the Ollama in a separate terminal:
   ```bash
      ollama serve
   ```
6. Run the dashboard:
   ```bash
   python app.py
   ```
