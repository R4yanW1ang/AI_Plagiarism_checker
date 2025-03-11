import torch
import re
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from accelerate.test_utils.testing import get_backend
import numpy as np

class PerplexityCalculator:
    def __init__(self, model_id="gpt2", ppl_threshold=17, sharpness=1, burstiness_weight=0.2):
        """Initialize the PerplexityCalculator class by loading the specified GPT-2 model and tokenizer onto the detected device."""
        self.device, _, _ = get_backend()
        self.model = GPT2LMHeadModel.from_pretrained(model_id).to(self.device)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
        self.max_length = self.model.config.n_positions
        self.stride = 512
        self.ppl_threshold = ppl_threshold  # Threshold for AI probability conversion
        self.sharpness = sharpness  # Controls steepness of sigmoid function
        self.burstiness_weight = burstiness_weight  # Weight for burstiness in AI probability calculation
    
    def split_sentences(self, text):
        """Splits text into sentences while handling empty ones."""
        sentences = re.split(r'(?<=[.?!])\s+(?=[A-Z])|\n+', text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]
    
    def compute_perplexity(self, text):
        """Compute the perplexity of a given text."""
        encodings = self.tokenizer(text, return_tensors="pt")
        seq_len = encodings.input_ids.size(1)
        
        if seq_len == 0:
            return float("inf")

        nll_sum = 0.0
        n_tokens = 0
        prev_end_loc = 0

        for begin_loc in tqdm(range(0, seq_len, self.stride)):
            end_loc = min(begin_loc + self.max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            
            if input_ids.numel() == 0:
                continue

            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss

            num_valid_tokens = (target_ids != -100).sum().item()
            num_loss_tokens = num_valid_tokens - input_ids.size(0)
            
            if num_loss_tokens > 0:
                nll_sum += neg_log_likelihood * num_loss_tokens
                n_tokens += num_loss_tokens

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        if n_tokens == 0:
            return float("inf")

        avg_nll = nll_sum / n_tokens
        return torch.exp(avg_nll).item()
    
    def analyze_text(self, text):
        """Compute perplexity for each sentence in the text."""
        sentences = self.split_sentences(text)
        results = [(sentence, self.compute_perplexity(sentence)) for sentence in sentences]
        results.sort(key=lambda x: x[1])  # Sort by perplexity (ascending)
        return results
    
    def compute_burstiness(self, sentence_lengths):
        """Compute the burstiness of a text using the coefficient of variation."""
        if len(sentence_lengths) < 2:
            return 0  # Not enough data to compute burstiness
        
        mean_length = np.mean(sentence_lengths)
        std_dev = np.std(sentence_lengths)
        burstiness = std_dev / mean_length if mean_length != 0 else 0

        return 1 - (burstiness / (1 + burstiness))
    
    def compute_ai_probability(self, text):
        """Compute AI probability based on perplexity and burstiness."""
        ppl = self.compute_perplexity(text)
        sentences = self.split_sentences(text)
        burstiness = self.compute_burstiness([len(sentence.split()) for sentence in sentences])
        
        ai_prob_ppl = 1 / (1 + np.exp((ppl - self.ppl_threshold) / self.sharpness))
        ai_prob = (1 - self.burstiness_weight) * ai_prob_ppl + self.burstiness_weight * burstiness
        
        return ppl, burstiness, ai_prob
    
    def preprocess_text(self, text):
        """Normalize line breaks within paragraphs before sentence splitting."""
        return re.sub(r'(?<!\n)\n(?!\n)', ' ', text)


if __name__ == "__main__":
    text = """
I had been using VSCode for a long time. It's easy to install and use, and for new language support, I just need to install the appropriate extensions from their store. It was a perfect tool for a beginner like me at that time.

However, as my workload grew heavier, managing large repos became necessary for smooth operation on my machine and maintaining consistency across different OS environments.

My job involves heavily using EC2 machines, SSH'ing into production servers, and working with pods. An environment with no display or graphical unit was one of the main reasons I decided to switch to something else. I aim to find an IDE that offers better performance, customizability, compatibility, and efficiency in text editing.

Vim is the first thing that comes to mind. Since the vi editor is a default tool on most operating systems, I became curious about what vi actually is. After researching vi, Vim, and NeoVim, I decided it was the right time to switch to this family — specifically NeoVim.

NeoVim is a lightweight text editor that offers lightning-fast performance, infinite customization options, and enhanced productivity for text editing. It also allows me to maintain a consistent, personalized development experience across different workstations. Despite its reasonably steep learning curve, NeoVim is well worth using.

Brief History

NeoVim is a fork of the original Vim, authored and maintained by Bram Moolenaar. His position in the Vim project is 'dictator for life,' meaning whatever he says goes, and whatever he doesn't say doesn't. As you can see in the contribution chart below, he makes almost all the commits in Vim. That being said, he has total control over Vim's direction.

Neovim was created because some people were frustrated with Bram Moolenaar's leadership. The project started in 2014 after a patch to Vim that supported multi-threading was rejected. Neovim had a successful fundraising campaign in March 2014, which supported at least one full-time developer. Several frontends are currently under development, which use Neovim's capabilities.

Vim and Neovim are similar, but one of the most significant differences is that Neovim has a much larger community, which tends to be more welcoming to new features. Beyond that, Vim and Neovim are pretty similar.
    """
    calculator = PerplexityCalculator()
    sentence_results = calculator.analyze_text(text)
    full_ppl, burstiness, ai_prob = calculator.compute_ai_probability(text)
    
    print(f"Full Perplexity: {full_ppl}")
    print(f"Burstiness: {burstiness}")
    print(f"AI Probability: {ai_prob}")
    print()
    print("\nTop 5 sentences with least perplexity:")
    for sentence, ppl in sentence_results[:5]:
        print(f"{round(ppl, 2)}: {sentence} ")
    print("\nTop 5 sentences with largest perplexity:")
    for sentence, ppl in sentence_results[-5:]:
        print(f"{round(ppl, 2)}: {sentence} ")