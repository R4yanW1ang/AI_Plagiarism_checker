import json
import torch
import os

class ResultGenerator:
    def __init__(self, assignment_id):
        """Initialize ResultGenerator class and set result directory."""
        self.assignment_id = assignment_id
        self.result_path = f'data/assignment_{assignment_id}/results'
        os.makedirs(self.result_path, exist_ok=True)
    
    def generate_results(self, sentence_results, full_ppl, full_burst, ai_prob, file_name):
        """Generate AI probability, perplexity, and sentence-level analysis based on given results."""
        # Extract top 5 highest and lowest perplexity sentences
        top_5_highest = sorted(sentence_results, key=lambda x: x[1], reverse=True)[:5]
        top_5_lowest = sorted(sentence_results, key=lambda x: x[1])[:5]

        print("ai_prob: ", ai_prob)
        print("full_ppl: ", full_ppl)
        print("full_burst: ", full_burst)
        print("top_5_highest: ", top_5_highest)
        print("top_5_lowest: ", top_5_lowest)

        results = {
            "ai_probability": float(ai_prob) if isinstance(ai_prob, torch.Tensor) else ai_prob,
            "perplexity": float(full_ppl) if isinstance(full_ppl, torch.Tensor) else full_ppl,
            "burstiness": float(full_burst) if isinstance(full_burst, torch.Tensor) else full_burst,
            "top_5_lowest_perplexity_sentences": [{"sentence": s, "perplexity": float(p) if isinstance(p, torch.Tensor) else p} for s, p in top_5_lowest],
            "top_5_highest_perplexity_sentences": [{"sentence": s, "perplexity": float(p) if isinstance(p, torch.Tensor) else p} for s, p in top_5_highest]
        }

        result_file = os.path.join(self.result_path, os.path.basename(file_name).replace('.txt', '_results.json'))
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        
        return results
