import os
import json
from config import Config
from fetch_data import CanvasSubmissionDownloader
from ai_detector import PerplexityCalculator
from result_generator import ResultGenerator

def main():
    # Load configuration
    config = Config()
    
    # Initialize Canvas Submission Downloader
    downloader = CanvasSubmissionDownloader(
        api_url=config.canvas_url,
        api_key=config.canvas_token,
        course_id=config.course_id,
        assignment_id=config.assignment_id
    )
    
    # Retrieve submissions
    downloader.get_assignment_submissions()
    
    # Initialize Perplexity Calculator
    calculator = PerplexityCalculator(
        model_id=config.openai_model_id,
        ppl_threshold=config.ppl_threshold,
        sharpness=config.ppl_sharpness,
        burstiness_weight=config.burstiness_weight
    )
    
    # Process each submission
    submission_folder = f'data/assignment_{config.assignment_id}/submissions'
    result_generator = ResultGenerator(config.assignment_id)
    
    # Iterate through each submission file
    for file_name in os.listdir(submission_folder):
        file_path = os.path.join(submission_folder, file_name)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Compute perplexity analysis
        sentence_results = calculator.analyze_text(text)
        full_ppl, burstiness, ai_prob = calculator.compute_ai_probability(text)

        # Generate results
        result_file = file_path.replace('.txt', '_results.json')
        results = result_generator.generate_results(sentence_results, full_ppl, burstiness, ai_prob, result_file)
        
        print(f"Processed {file_name}, results saved to {result_file}")
    
if __name__ == "__main__":
    main()