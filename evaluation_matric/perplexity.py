import math
import unittest


def calculate_perplexity(labels, predictions):
    """
    Calculate perplexity between label words and predicted words.

    Parameters:
    - labels: List of label words (list of strings)
    - predictions: List of predicted words (list of strings)

    Returns:
    - Perplexity (float)
    """
    # Ensure the lengths of labels and predictions are the same
    if len(labels) != len(predictions):
        raise ValueError("The lengths of labels and predictions must be the same.")

    # Calculate cross-entropy
    cross_entropy = 0.0
    for label in labels:
        # Assuming a simple model where the probability of each word is independent
        word_prob = predictions.get(label, 1e-10)  # Avoid zero probability
        cross_entropy += -math.log(word_prob)

    # Calculate perplexity
    perplexity = math.exp(cross_entropy / len(labels))

    return perplexity


class PerplexityTest(unittest.TestCase):

    def test_bleu_score_calculation(self):
        label_words = ["the", "quick", "brown", "fox"]
        predicted_probabilities = {"the": 0.5, "quick": 0.3, "brown": 0.2, "fox": 0.1}

        perplexity_value = calculate_perplexity(label_words, predicted_probabilities)
        print(f"Perplexity: {perplexity_value}")


if __name__ == "__main__":
    unittest.main()
