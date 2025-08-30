def get_chatbot_response(question, spending_history, forecast, transactions, user_info):
    """Generate a response to a user's question about their finances.

    Args:
        user_id: The ID of the user
        question: The user's question
        spending_history: The user's spending history
        forecast: The user's spending forecast
        transactions: The user's transactions
        user_info: The user's information

    Returns:
        str: The chatbot's response
    """
    # Convert question to lowercase for easier matching
    question_lower = question.lower()

    # Check if spending history is available
    has_history = not spending_history.empty

    # Check for different types of questions
    if any(keyword in question_lower for keyword in ['hello', 'hi', 'hey', 'greetings']):
        return f"Hello {user_info['name']}! How can I help you with your finances today?"

    elif any(keyword in question_lower for keyword in ['income', 'earn', 'salary', 'make']):
        return f"Your average monthly income is Rs. {user_info['income']:,.2f}."

    elif any(keyword in question_lower for keyword in ['spend', 'spending', 'expenses']):
        if has_history:
            return f"Your average monthly spending is Rs. {user_info['expenses']:,.2f}."
        else:
            return "I don't have enough spending history to answer that question."

    elif any(keyword in question_lower for keyword in ['save', 'saving', 'savings']):
        if has_history:
            return f"Based on your income and spending, you save approximately Rs. {user_info['savings']:,.2f} per month, which is {user_info['savings'] / user_info['income'] * 100:.1f}% of your income."
        else:
            return "I don't have enough spending history to calculate your savings."

    elif any(keyword in question_lower for keyword in ['budget', 'recommend', 'suggestion', 'advice']):
        if has_history:
            # Calculate spending by category
            total_by_category = spending_history.groupby("Category")["Expense"].sum().sort_values(ascending=False)
            # Find the top spending category
            top_category = total_by_category.idxmax()
            top_amount = total_by_category.max()
            # Calculate percentage of income
            top_percent = (top_amount / user_info['income']) * 100
            top_n = total_by_category.head(5)

            return f"Your highest spending category is {top_category}, where you spend an average of Rs. {top_amount/len(spending_history):,.2f} per month ({top_percent:.1f}% of your income). Your top 5 spending categories are: {list(top_n.index)}."
        else:
            return "I don't have enough spending history to provide budget recommendations."

    elif any(keyword in question_lower for keyword in ['forecast', 'predict', 'future', 'next month']):
        if not forecast.empty:
            next_month_total = forecast.iloc[0].sum()
            return f"Based on your spending history, I forecast that you'll spend approximately ${next_month_total:,.2f} next month."
        else:
            return "I don't have enough data to make a forecast."

    elif any(keyword in question_lower for keyword in ['transaction', 'recent', 'last purchase']):
        if not transactions.empty:
            latest_transaction = transactions.iloc[0]
            return f"Your most recent transaction was ${latest_transaction['amount']:.2f} for {latest_transaction['category']} on {latest_transaction['transaction_date'].strftime('%Y-%m-%d')}."
        else:
            return "I don't have any transaction data for you."

    elif any(keyword in question_lower for keyword in ['help', 'can you do', 'what can you do', 'features']):
        return ("I can help you with:\n"
                "- Information about your income and spending\n"
                "- Savings analysis\n"
                "- Budget recommendations\n"
                "- Spending forecasts\n"
                "- Recent transactions\n\n"
                "Just ask me a question about any of these topics!")

    else:
        return "I'm not sure how to answer that question. You can ask me about your income, spending, savings, budget recommendations, forecasts, or recent transactions."
