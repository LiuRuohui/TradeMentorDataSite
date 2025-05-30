import os
import openai
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
import re
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="TradeMentor.log",
    filemode="a"
)


test_data = {"test_type": None, "user_info": {}}

load_dotenv()

#Configure environment
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_ORGANIZATION = os.getenv('OPENAI_ORGANIZATION')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL')
MODEL_NAME = os.getenv('MODEL_NAME')


if not OPENAI_API_KEY:
    raise ValueError("API key is not set in the environment variables.")
if not MODEL_NAME:
    raise ValueError("Model name is not set in the environment variables.")

# Configure OpenAI API
openai.api_key = OPENAI_API_KEY
if OPENAI_BASE_URL:
    openai.api_base = OPENAI_BASE_URL
if OPENAI_ORGANIZATION:
    openai.organization = OPENAI_ORGANIZATION

# Initialize the flask application
app = Flask(__name__)

# Process user input and return chat response
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form["user_input"]
    chatbot_type = request.form['chatbot_type']
    duration = int(request.form['duration'])

    logging.info(f"User Input: {user_input}, Chatbot Type: {chatbot_type}, Duration: {duration} seconds")

    # Adjust system message based on chatbot_type
    chatbot_tones = {
        'humorous': "This chatbot will respond with jokes and lighthearted comments.",
        'professional': "This chatbot will respond in a formal and business-appropriate manner.",
        'educational': "This chatbot will focus on teaching and explaining concepts.",
        'supportive': "This chatbot will provide empathetic and encouraging responses."
    }

    # Update system message based on selected chatbot type
    system_message = f"Your role is a professional trade consultant. {chatbot_tones.get(chatbot_type, 'This chatbot will respond in a helpful manner.')}"
    
    # Modify the system message dynamically
    messages[0] = {"role": "system", "content": system_message}

    # Add user input to dialog history
    messages.append({"role": "user", "content": user_input})

    try:
        # Generate assistant response using openai API
        response = openai.chat.completions.create(
            model=MODEL_NAME,
            messages=messages
        )
        assistant_reply = response.choices[0].message.content
        assistant_reply = auto_segment(assistant_reply)

        # Add assistant's reply to conversation history
        messages.append({"role": "assistant", "content": assistant_reply})

        # Record assistant response to log
        logging.info(f"助手回应: {assistant_reply}")

        return jsonify({"assistant_reply": assistant_reply})

    except Exception as e:
        logging.error(f"生成助手回应时出错: {e}")
        return jsonify({"error": str(e)})

# Initialize dialog history
messages = [
    {"role": "system", "content": 
    "Your role is a professional trade consultant.Welcome to the TradeMentor! As a virtual assistant, I am here to help you navigate the basics of stock trading and answer your questions. "
    "Adjust different tones according to the type selected by the user. Humorous：This chatbot will respond with jokes and lighthearted comments；Professional：This chatbot will respond in a formal and business-appropriate manner；Educational：This chatbot will focus on teaching and explaining concepts；Supportive：This chatbot will provide empathetic and encouraging responses"  }
]
def auto_segment(text):
    segments = re.split(r'(?<=\.|\!|\?)\s+', text.strip())
    segmented_text = "\n\n".join(segments)
    return segmented_text

# Home page routing, rendering HTML pages
@app.route("/")
def home():
      return render_template("index.html")

@app.route('/submit_form', methods=['POST'])
def submit_form():
    data = request.get_json()
    testing_number = data.get('testing_number')
    age = data.get('age')
    gender = data.get('gender')

    logging.info(f"Received: Testing Number={testing_number}, Age={age}, Gender={gender}")

    return jsonify({"message": "Form submitted successfully!"}), 200

@app.route("/end-service", methods=["POST"])
def end_service():
    chatbot_type = request.form['chatbot_type']
    total_duration = int(request.form['total_duration'])

    # Record total duration
    logging.info(f"Chatbot Type: {chatbot_type}, Total Duration: {total_duration} seconds")

    return jsonify({"message": "Service ended and duration recorded."})

@app.route("/submit_user_info", methods=["POST"])
def submit_user_info():
    user_info = request.form.to_dict()
    test_data["user_info"] = user_info
    logging.info(f"用户提交的信息: {user_info}")
    return jsonify({"status": "success", "message": "用户信息已提交"})
# Start the flask application
if __name__ == "__main__":
    app.run(debug=True, port=5001)
