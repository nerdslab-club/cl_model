import math
import unittest
from typing import Any


from category_router.category_router import CategoryRouter
from cl_pretrainer.pre_trainer_utils import PreTrainerUtils
from vocabulary_builder.output_vocabulary_builder import OutputVocabBuilder


def calculate_sentence_perplexity(labels, predictions):
    """Calculate perplexity between label words and predicted words.

    :param labels: List of label words (list of strings)
    :param predictions: Dict of predicted words with prediction value
    :return: Perplexity (float)
    """
    # Ensure the lengths of labels and predictions are the same
    all_keys_present, missing_keys = check_keys_presence(labels, predictions)
    if not all_keys_present:
        print(f"Following keys aren't present for the label: {labels} and missing_keys: {missing_keys}")

    # Calculate cross-entropy
    cross_entropy = 0.0
    for label in labels:
        # Assuming a simple model where the probability of each word is independent
        word_prob = predictions.get(label, 1e-10)  # Avoid zero probability
        cross_entropy += -math.log(word_prob)

    # Calculate perplexity
    perplexity = math.exp(cross_entropy / len(labels))

    return perplexity


def calculate_batch_perplexity(batch_labels, batch_predictions):
    """Calculate average perplexity for a batch of sentences.

    :param batch_labels: List of lists, where each inner list contains label words for a sentence
    :param batch_predictions: List of dicts, where each inner dict contains predicted words with values for a sentence
    :return: Average Perplexity (float)
    """
    # Ensure the lengths of batch_labels and batch_predictions are the same
    if len(batch_labels) != len(batch_predictions):
        print(f"length of batch_labels is {len(batch_labels)}")
        print(f"length of batch_predictions is {len(batch_predictions)}")
        raise ValueError("The lengths of batch_labels and batch_predictions must be the same.")

    total_perplexity = 0.0

    for labels, predictions in zip(batch_labels, batch_predictions):
        perplexity = calculate_sentence_perplexity(labels, predictions)
        total_perplexity += perplexity

    # Calculate average perplexity for the batch
    average_perplexity = total_perplexity / len(batch_labels)

    return average_perplexity


def get_target_tokens_probability(
        target_batch: list[list[dict]],
        output_logits_map: dict[int, dict[str, Any]],
        output_vocab_builder: OutputVocabBuilder,
) -> list[dict]:
    batch_predicted_probabilities = []

    batch_route_and_token_ids: list[list[tuple[int, int]]] = output_vocab_builder.batch_encoder(target_batch, False)
    batch_extracted_tokens = PreTrainerUtils.extract_tokens(target_batch)

    for sentence_index, sentence_route_ids in enumerate(batch_route_and_token_ids):
        sentence_predicted_probabilities = {}
        for word_index, route_and_token_id in enumerate(sentence_route_ids):
            route_id, token_index = route_and_token_id
            current_output_logits_map = output_logits_map[route_id]
            current_probabilities = current_output_logits_map[CategoryRouter.SOFTMAX_PROBABILITY]

            extracted_token = batch_extracted_tokens[sentence_index][word_index]
            sentence_predicted_probabilities[extracted_token] = current_probabilities[sentence_index][word_index][token_index].item()

        batch_predicted_probabilities.append(sentence_predicted_probabilities)

    print(batch_predicted_probabilities)
    return batch_predicted_probabilities


def check_keys_presence(keys: list, my_dict: dict):
    missing_keys = [key for key in keys if key not in my_dict]
    return len(missing_keys) == 0, missing_keys


class PerplexityTest(unittest.TestCase):

    def test_sentence_perplexity_score_calculation(self):
        label_words = ["the", "quick", "brown", "fox", "joaa"]
        predicted_probabilities = {"the": 0.5, "quick": 0.3, "brown": 0.2, "fox": 0.1}

        perplexity_value = calculate_sentence_perplexity(label_words, predicted_probabilities)
        print(f"Sentence Perplexity: {perplexity_value}")
        self.assertAlmostEqual(319.57717183806074, perplexity_value)

    def test_corpus_perplexity_score_calculation(self):
        batch_labels = [["the", "quick", "brown", "fox"], ["the", "quick", "brown", "fox"]]
        batch_predictions = [
            {"the": 0.5, "quick": 0.3, "brown": 0.5, "fox": 0.1},
            {"the": 0.5, "quick": 0.3, "brown": 0.2, "fox": 0.1},
            ]

        perplexity_value = calculate_batch_perplexity(batch_labels, batch_predictions)
        print(f"Corpus Perplexity: {perplexity_value}")
        self.assertAlmostEqual(3.83547927, perplexity_value)


if __name__ == "__main__":
    unittest.main()
