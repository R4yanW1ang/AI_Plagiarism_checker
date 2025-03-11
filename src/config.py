import json
import os

class Config:
    def __init__(self, config_path='config.json'):
        """Initialize configuration by loading from a JSON file."""
        self.load_config(config_path)
    
    def load_config(self, config_path):
        """Load configurations from the specified JSON file."""
        with open(config_path, 'r') as f:
            config = json.load(f)

            # Load Canvas configurations
            canvas_config = config.get('canvas', {})
            self.canvas_token = os.getenv('CANVAS_TOKEN', canvas_config.get('token'))
            self.canvas_url = canvas_config.get('url')
            self.course_id = canvas_config.get('COURSE_ID')
            self.assignment_id = canvas_config.get('ASSIGNMENT_ID')

            # Load OpenAI configurations
            openai_config = config.get('openai', {})
            self.openai_model_id = openai_config.get('model_id', 'gpt2')
            self.ppl_threshold = openai_config.get('threshold', 20)
            self.ppl_sharpness = openai_config.get('sharpness', 5)
            self.burstiness_weight = openai_config.get('burstiness_weight', 0.2)

# Example usage
if __name__ == "__main__":
    config = Config()
    print(f"Canvas URL: {config.canvas_url}")
    print(f"OpenAI Model: {config.openai_model_id}")