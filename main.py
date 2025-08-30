import streamlit as st
from utils.chatbot import get_chatbot_response
from utils.data_processing import load_data, preprocess_data
from utils.classification import predict_expense_category
from utils.llm_assistant import OllamaAssistant
from utils.regression import recommend_budget
from utils.forecasting import forecast_expenses

st.title("Personal Finance AI Advisor")

expense_data = load_data("data/expense_descriptions.csv")
budget_data = load_data("data/income_saving_expense.csv")
# forecast_data = load_data("data/monthly_expenses.csv")
spending_history = load_data("data/category_date_expense.csv")

# Initialize Ollama assistant
ollama_assistant = OllamaAssistant()

# Check if Ollama is available
ollama_available = ollama_assistant.check_ollama_availability()

if ollama_available:
    st.sidebar.success("Ollama LLM is available")

    # Get available models
    available_models = ollama_assistant.get_available_models()

    if available_models:
        # Model selection
        selected_model = st.sidebar.selectbox(
            "Select LLM Model",
            options=available_models,
            index=0
        )

        # Update the model in the assistant
        ollama_assistant.model_name = selected_model

        # Temperature setting
        temperature = st.sidebar.slider(
            "Temperature",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher values make the output more random, lower values make it more deterministic"
        )
    else:
        st.sidebar.warning("No models available. Please pull a model using Ollama CLI.")
        selected_model = None
        temperature = 0.7
else:
    st.sidebar.error("Ollama LLM is not available. Make sure it's running at http://localhost:11434")
    selected_model = None
    temperature = 0.7

option = st.sidebar.selectbox(
    "Choose a feature:",
    ["Expense Categorization", "Budget Recommendations", "Expense Forecasting", "Financial Assistance"]
)

if option == "Expense Categorization":
    st.header("Predictive Expense Categorization")
    description = st.text_input("Enter expense description:")
    if st.button("Classify Expense"):
        category = predict_expense_category(description)
        st.write(f"Predicted Category: **{category}**")

elif option == "Budget Recommendations":
    st.header("Budget Recommendations")
    income = st.number_input("Enter your monthly income:")
    if st.button("Get Budget Recommendation"):
        breakdown = recommend_budget(income, budget_data)
        st.write("### Data-Driven Budget")
        for k, v in breakdown.items():
            st.write(f"- {k}: Rs.{v:,.2f}")

elif option == "Expense Forecasting":
    st.header("Expense Forecasting")
    months = st.number_input("Enter months to forecast:", min_value=1, max_value=12)
    if st.button("Forecast Expenses"):
        forecast = forecast_expenses(budget_data, months)
        st.line_chart(forecast)
 
elif option == "Financial Assistance":
   
 # st.header("Financial Assistance")
    # st.write("Ask me anything about your finances and I'll try to help you.")

    # # Initialize chat history in session state if it doesn't exist
    # if 'chat_history' not in st.session_state:
    #     st.session_state.chat_history = []

    # # Display chat history
    # for message in st.session_state.chat_history:
    #     if message['is_user']:
    #         st.chat_message("user").write(message['text'])
    #     else:
    #         st.chat_message("assistant").write(message['text'])

    # # Chat input
    # user_question = st.chat_input("Ask a question about your finances...")

    # if user_question:
    #     # Add user message to chat history
    #     st.session_state.chat_history.append({"is_user": True, "text": user_question})

    #     # Display user message
    #     st.chat_message("user").write(user_question)

    #     forecast = forecast_expenses(budget_data, 12)
    #     avg_saving = budget_data['Saving'].mean()
    #     avg_income = budget_data['Income'].mean()
    #     avg_expenses = budget_data['Expenses'].mean()

    #     user_info = {
    #         "name": "User",
    #         "income": avg_income,
    #         "expenses": avg_expenses,
    #         "savings": avg_saving
    #     }


    #     # Generate response
    #     with st.spinner("Thinking..."):
    #         response = get_chatbot_response(
    #             user_question,
    #             spending_history,
    #             forecast,
    #             budget_data,
    #             user_info
    #         )

    #     # Add assistant response to chat history
    #     st.session_state.chat_history.append({"is_user": False, "text": response})

    #     # Display assistant response
    #     st.chat_message("assistant").write(response)

    # # Add a button to clear chat history
    # if st.session_state.chat_history and st.button("Clear Chat History"):
    #     st.session_state.chat_history = []
    #     st.rerun()
    st.subheader("Chat with Finance Assistant")
    st.write("Ask me anything about your finances and I'll try to help you.")

    # Add a welcome note with examples of what the chatbot can do
    welcome_note = """
    💬 **Welcome to your Finance Assistant!**

    I can help you with:
    - Your income and spending patterns
    - Savings analysis and advice
    - Budget recommendations
    - Spending forecasts
    - Recent transactions
    - Specific spending categories (e.g., 'How much do I spend on food?')
    - Spending over different time periods (daily, weekly, monthly, yearly)

    **Try asking me questions like:**
    - "What's my monthly income?"
    - "How much do I spend each month?"
    - "What's my savings rate?"
    - "Give me budget recommendations"
    - "What's my forecast for next month?"
    - "Show me my recent transactions"
    - "How much do I spend on groceries?"
    - "What's my annual spending?"
    """
    st.markdown(f'<div class="welcome-note">{welcome_note}</div>', unsafe_allow_html=True)

    # Create columns for chat interface layout
    chat_col1, chat_col2 = st.columns([8, 2])

    # Initialize chat history in session state if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        # Add a welcome message from the assistant
        st.session_state.chat_history.append({"is_user": False, "text": "Hi there! I'm your Finance Assistant. How can I help you today?"})

    # Create a container for the chat messages with custom styling
    with chat_col1:
        chat_container = st.container()
        with chat_container:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)

            # Display chat history with custom styling
            for message in st.session_state.chat_history:
                if message['is_user']:
                    st.markdown(f'<div class="chat-message-user">{message["text"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message-assistant">{message["text"]}</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    # Add buttons in the second column
    with chat_col2:
        # Add a button to clear chat history
        if st.session_state.chat_history and st.button("Clear Chat"):
            # Keep only the welcome message
            st.session_state.chat_history = [st.session_state.chat_history[0]]
            st.rerun()

        # Add some example quick questions as buttons
        st.markdown("### Quick Questions")
        if st.button("My Income"):
            user_question = "What's my monthly income?"
            st.session_state.quick_question = user_question
            st.rerun()

        if st.button("My Spending"):
            user_question = "How much do I spend each month?"
            st.session_state.quick_question = user_question
            st.rerun()

        if st.button("Budget Tips"):
            user_question = "Give me budget recommendations"
            st.session_state.quick_question = user_question
            st.rerun()

        if st.button("Forecast"):
            user_question = "What's my forecast for next month?"
            st.session_state.quick_question = user_question
            st.rerun()

    # Chat input at the bottom
    user_question = st.chat_input("Ask a question about your finances...", key="chat_input")

    # Handle quick question buttons
    if 'quick_question' in st.session_state:
        user_question = st.session_state.quick_question
        del st.session_state.quick_question

    if user_question:
        # Add user message to chat history
        st.session_state.chat_history.append({"is_user": True, "text": user_question})

        forecast = forecast_expenses(budget_data, 12)
        budget_recommendations = recommend_budget(forecast.iloc[0], budget_data)
        avg_saving = budget_data['Saving'].mean()
        avg_income = budget_data['Income'].mean()
        avg_expenses = budget_data['Expense'].mean()

        user_info = {
            "name": "User",
            "income": avg_income,
            "expenses": avg_expenses,
            "savings": avg_saving
        }
        # Generate response
        with st.spinner("Thinking..."):
            # Prepare financial data for the LLM
            financial_data = {
                "user_info": dict(user_info),
                "spending_history": spending_history,
                "forecast": forecast,
                "overall": budget_data,
                "budget_recommendations": budget_recommendations
            }

            # Use Ollama LLM if available, otherwise fall back to rule-based responses
            if ollama_available and selected_model:
                response = ollama_assistant.generate_response(
                    user_question,
                    financial_data,
                    temperature=temperature
                )
            else:
                # Fall back to rule-based responses if Ollama is not available
                response = get_chatbot_response(
                    user_question,
                    spending_history,
                    forecast,
                    budget_data,
                    user_info
                )

        # Add assistant response to chat history
        st.session_state.chat_history.append({"is_user": False, "text": response})

        # Rerun to update the UI
        st.rerun()