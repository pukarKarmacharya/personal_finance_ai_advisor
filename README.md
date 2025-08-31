# Personal Finance Agentic Advisor

This project is a **personal finance advisor** that uses machine learning and time series forecasting to help users manage their money. It automates expense categorization, predicts budgets, forecasts income/savings/expenses, and supports an interactive chatbot interface.

---

## Features

- **Expense Categorization**  
  Uses TF-IDF and Logistic Regression to classify transaction descriptions into categories like food, rent, or travel.

- **Budget Recommendation**  
  Predicts expenses from income and savings using Ridge/Lasso regression. Splits results into needs, wants, and savings.

- **Financial Forecasting**  
  Uses Prophet models to forecast income, savings, and expenses. Accounts for Nepal holidays to capture seasonal effects.

- **Conversational Assistant**  
  Chat layer powered by Ollama (LLM) when available, or a rule-based system as fallback. Users can ask natural language questions about their finances.

- **Interactive Interface**  
  Built with Streamlit to provide an easy-to-use dashboard with charts, tables, and chatbot access.

---

## Installation

1. Clone the repository
2. `uv sync`
3. `streamlit run main.py`