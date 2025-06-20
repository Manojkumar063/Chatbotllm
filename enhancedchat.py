import openai
import json
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChatBot:
    def __init__(self, api_key, default_model="gpt-3.5-turbo", history_file="chat_history.json"):
        """Initialize the chatbot with API key and configuration."""
        openai.api_key = api_key
        self.model = default_model
        self.history_file = history_file
        self.conversation_history = []
        self.system_prompt = "You are a helpful assistant with expertise in various topics."
        self.temperature = 0.7
        self.max_tokens = 1000
        self.load_history()

    def load_history(self):
        """Load conversation history from file if it exists."""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    self.conversation_history = json.load(f)
                logger.info("Loaded conversation history from %s", self.history_file)
        except Exception as e:
            logger.error("Failed to load history: %s", str(e))

    def save_history(self):
        """Save conversation history to file."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.conversation_history, f, indent=2)
            logger.info("Saved conversation history to %s", self.history_file)
        except Exception as e:
            logger.error("Failed to save history: %s", str(e))

    def chat(self, user_input, model=None):
        """Process user input and get response from OpenAI API."""
        try:
            if model:
                self.model = model

            # Add user message to history
            self.conversation_history.append({"role": "user", "content": user_input})

            # Prepare messages for API call
            messages = [
                {"role": "system", "content": self.system_prompt},
                *self.conversation_history[-5:]  # Include last 5 messages for context
            ]

            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            assistant_response = response['choices'][0]['message']['content'].strip()
            self.conversation_history.append({"role": "assistant", "content": assistant_response})
            self.save_history()
            return assistant_response

        except Exception as e:
            logger.error("API call failed: %s", str(e))
            return f"Error: {str(e)}"

    def set_system_prompt(self, prompt):
        """Update the system prompt."""
        self.system_prompt = prompt
        logger.info("System prompt updated")

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        if os.path.exists(self.history_file):
            os.remove(self.history_file)
        logger.info("Conversation history cleared")

def main():
    # Replace with your actual API key
    api_key = "your-api-key-here"
    
    # Available models
    available_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
    
    # Initialize chatbot
    bot = ChatBot(api_key)
    
    print("Welcome to the Enhanced ChatBot!")
    print("Available commands:")
    print("- /exit or /quit: Exit the program")
    print("- /clear: Clear conversation history")
    print("- /model <model_name>: Switch model (available: " + ", ".join(available_models) + ")")
    print("- /prompt <new_prompt>: Set new system prompt")
    print("\nStart typing to chat!\n")

    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['/exit', '/quit']:
                print("Goodbye!")
                break
                
            elif user_input.lower() == '/clear':
                bot.clear_history()
                print("Conversation history cleared!")
                
            elif user_input.lower().startswith('/model'):
                try:
                    _, model = user_input.split(maxsplit=1)
                    if model in available_models:
                        print(f"Switching to model: {model}")
                    else:
                        print(f"Invalid model. Available models: {', '.join(available_models)}")
                except ValueError:
                    print("Usage: /model <model_name>")
                    
            elif user_input.lower().startswith('/prompt'):
                try:
                    _, new_prompt = user_input.split(maxsplit=1)
                    bot.set_system_prompt(new_prompt)
                    print("System prompt updated!")
                except ValueError:
                    print("Usage: /prompt <new_prompt>")
                    
            else:
                response = bot.chat(user_input)
                print("LLM:", response)
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error("Error in main loop: %s", str(e))
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
