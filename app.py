import gradio as gr
import pandas as pd
import pickle
import numpy as np
import os

# Load the trained model
model_path = 'gb_churn_model.pkl'

if not os.path.exists(model_path):
    # If the model doesn't exist, we might need to tell the user to run the training script
    # However, since the user asked to make the interface, I'll assume they want it ready to run.
    # For now, I'll just raise an error if it's missing when launched.
    pass

def predict_churn(credit_score, geography, gender, age, tenure, balance, num_products, has_card, is_active, salary):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
            
        # Map Yes/No strings to 1/0
        has_card_val = 1 if has_card == "Yes" else 0
        is_active_val = 1 if is_active == "Yes" else 0
            
        # Create a DataFrame for prediction matching the training feature names and types
        input_data = pd.DataFrame([{
            'CreditScore': int(credit_score),
            'Geography': geography,
            'Gender': gender,
            'Age': int(age),
            'Tenure': int(tenure),
            'Balance': float(balance),
            'NumOfProducts': int(num_products),
            'HasCrCard': has_card_val,
            'IsActiveMember': is_active_val,
            'EstimatedSalary': float(salary)
        }])
        
        # Get prediction and probability
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        if prediction == 1:
            result = "🔴 The customer is likely to CHURN (Exited)."
            prob_text = f"Churn Probability: {probability:.2%}"
        else:
            result = "🟢 The customer is likely to STAY (Not Exited)."
            prob_text = f"Stay Probability: {(1-probability):.2%}"
            
        return f"{result}\n\nConfidence: {prob_text}"
    except Exception as e:
        return f"Error: {str(e)}"

# Custom CSS for a more premium look
custom_css = """
.gradio-container {
    font-family: 'Inter', sans-serif;
}
.main-title {
    text-align: center;
    color: #2D3748;
    margin-bottom: 2rem;
}
.predict-btn {
    background: linear-gradient(90deg, #3182ce 0%, #2c5282 100%) !important;
    border: none !important;
    color: white !important;
}
"""

# Create Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown(
        """
        <div class="main-title">
            <h1>🏦 Bank Customer Churn Predictor</h1>
            <p>Enter customer details below to predict the likelihood of them leaving the bank.</p>
        </div>
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 👤 Customer Profile")
            credit_score = gr.Slider(minimum=300, maximum=850, step=1, label="Credit Score", value=650)
            geography = gr.Dropdown(choices=["France", "Germany", "Spain"], label="Geography", value="France")
            gender = gr.Dropdown(choices=["Female", "Male"], label="Gender", value="Female")
            age = gr.Slider(minimum=18, maximum=100, step=1, label="Age", value=38)
            tenure = gr.Slider(minimum=0, maximum=10, step=1, label="Tenure (Years at Bank)", value=5)
            
        with gr.Column(scale=1):
            gr.Markdown("### 💰 Financial Status")
            balance = gr.Number(label="Account Balance", value=0.0, step=0.01)
            num_products = gr.Slider(minimum=1, maximum=4, step=1, label="Number of Products Purchased", value=1)
            has_card = gr.Radio(choices=["Yes", "No"], label="Has Credit Card?", value="Yes")
            is_active = gr.Radio(choices=["Yes", "No"], label="Is Active Member?", value="Yes")
            salary = gr.Number(label="Estimated Annual Salary", value=50000.0, step=0.01)

    with gr.Row():
        predict_btn = gr.Button("Analyze Customer Churn Risk", variant="primary", elem_classes="predict-btn")
    
    with gr.Row():
        output = gr.Textbox(label="Analysis Result", interactive=False)
    
    predict_btn.click(
        fn=predict_churn,
        inputs=[credit_score, geography, gender, age, tenure, balance, num_products, has_card, is_active, salary],
        outputs=output
    )
    
    gr.Markdown(
        """
        ---
        **Note:** This model uses Gradient Boosting trained on churn modelling data to provide predictions.
        """
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="blue", spacing_size="md", radius_size="lg"), css=custom_css)
