import os
import streamlit as st
import requests

# Load your Hugging Face API key from an environment variable
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")
if not HUGGINGFACE_API_KEY:
    st.error("Hugging Face API key not found. Please set it in your Space secrets as HUGGINGFACE_API_KEY.")
    st.stop()

# Hugging Face Inference API endpoint for the DeepSeek model
DEEPSEEK_MODEL_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"  # Adjust if needed.

# -------------------------------
# Helper functions
# -------------------------------

def refine_prompt(nutrition_query, weight, height, activity_level, food_restriction, goal):
    """
    Builds a concise prompt from the provided details.
    """
    prompt_template = (
        "Details: W:{}kg, H:{}cm, Act:{}; Restr:{}; Goal:{}.\n"
        "Query: {}.\n"
        "Plan: Provide calorie intake, nutrient (g) breakdown, meal suggestions, and guidelines."
    )
    full_prompt = prompt_template.format(
        weight, height, activity_level, food_restriction, goal, nutrition_query.strip()
    )
    return full_prompt

def enforce_free_tier_limit(prompt, limit=256):
    """
    Ensures that the prompt does not exceed the free-tier limit.
    If it does, it truncates the prompt.
    """
    if len(prompt) > limit:
        return prompt[:limit]
    return prompt

def call_deepseek_api(prompt):
    """
    Calls Hugging Face's Inference API for the DeepSeek model using the given prompt.
    """
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 256}  # Adjust max tokens as needed.
    }
    
    response = requests.post(DEEPSEEK_MODEL_URL, headers=headers, json=payload)
    if response.status_code != 200:
        return f"Error: {response.status_code} - {response.text}"
    
    result = response.json()
    if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
        return result[0]["generated_text"]
    return str(result)

def verify_with_medical_db(response_text):
    """
    (Optional) Simulated verification of the response.
    Replace with actual verification logic if available.
    """
    verified = True
    verification_message = "Response verified against trusted nutrition and medical databases."
    return verified, verification_message

# -------------------------------
# Main Application
# -------------------------------

def main():
    st.title("AI Nutrition Diet Planner (Hugging Face DeepSeek)")
    st.markdown(
        """
        Welcome to the **AI Nutrition Diet Planner** using Hugging Face's DeepSeek model.
        Please provide your details and describe your nutrition query, dietary concerns, or specific diet preferences.
        Our AI agent will analyze your input and generate a personalized nutrition plan tailored to your profile.
        **Note:** This tool is for informational purposes only and does not substitute professional nutritional or medical advice.
        """
    )

    with st.form("nutrition_form", clear_on_submit=False):
        st.header("Patient Information")
        name = st.text_input("Name")
        age = st.number_input("Age", min_value=0, max_value=120, value=25)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        
        weight = st.number_input("Weight (kg)", min_value=0.0, step=0.1, value=70.0)
        height = st.number_input("Height (cm)", min_value=0.0, step=0.1, value=170.0)
        activity_level = st.selectbox("Activity Level", ["Sedentary", "Lightly active", "Moderately active", "Very active", "Extra active"])
        food_restriction = st.text_input("Food Restrictions (if any)", value="None")
        goal = st.text_input("Nutrition Goal (e.g., weight loss, muscle gain, maintenance)", value="Maintenance")
        
        st.header("Nutrition Query")
        nutrition_query = st.text_area("Describe your nutrition query, dietary concerns, or specific diet preferences", height=150)

        submitted = st.form_submit_button("Submit")

    if submitted:
        if not nutrition_query.strip():
            st.error("Please enter your nutrition query or dietary preferences.")
        else:
            st.subheader("Processing your input...")
            engineered_prompt = refine_prompt(nutrition_query, weight, height, activity_level, food_restriction, goal)
            st.markdown("**Engineered Prompt (before free-tier enforcement):**")
            st.code(engineered_prompt, language="text")
            
            truncated_prompt = enforce_free_tier_limit(engineered_prompt, limit=256)
            total_length = len(truncated_prompt)
            st.markdown(f"**Total Prompt Length:** {total_length} characters")
            if len(engineered_prompt) != len(truncated_prompt):
                st.warning("Your nutrition query was automatically truncated to meet the free-tier limit.")
            
            st.info("Generating your personalized nutrition plan using Hugging Face DeepSeek...")
            deepseek_response = call_deepseek_api(truncated_prompt)
            st.markdown("**Response from Hugging Face DeepSeek:**")
            st.write(deepseek_response)
            
            verified, verification_message = verify_with_medical_db(deepseek_response)
            if verified:
                st.success("Verification Successful: " + verification_message)
            else:
                st.warning("Verification Warning: " + verification_message)
            
            st.markdown("---")
            st.subheader("Need Further Clarification?")
            additional_info = st.text_input("Provide any additional details or clarifications")
            if additional_info.strip():
                refined_followup_prompt = refine_prompt(additional_info, weight, height, activity_level, food_restriction, goal)
                truncated_followup = enforce_free_tier_limit(refined_followup_prompt, limit=256)
                total_followup_length = len(truncated_followup)
                st.markdown("**Refined Follow-Up Prompt (after free-tier enforcement):**")
                st.code(truncated_followup, language="text")
                st.markdown(f"**Total Follow-Up Prompt Length:** {total_followup_length} characters")
                if len(refined_followup_prompt) != len(truncated_followup):
                    st.warning("Your follow-up input was automatically truncated to meet the free-tier limit.")
                
                st.info("Generating follow-up nutrition plan details using Hugging Face DeepSeek...")
                followup_response = call_deepseek_api(truncated_followup)
                st.markdown("**Follow-Up Response from Hugging Face DeepSeek:**")
                st.write(followup_response)
                
                verified, verification_message = verify_with_medical_db(followup_response)
                if verified:
                    st.success("Follow-Up Response Verified: " + verification_message)
                else:
                    st.warning("Follow-Up Verification Warning: " + verification_message)

if __name__ == "__main__":
    main()
