import unittest

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def calculate_bleu_score(reference, candidate, bleu_weights):
    """Calculate BLEU score between reference and candidate sentences.

    :param reference: List of reference words (list of strings)
    :param candidate: List of candidate words (list of strings)
    :param bleu_weights: How manu n-grams to consider.
    :return: BLEU score (float)
    """
    # Convert the reference and candidate sentences into a list of lists of tokens
    reference_tokens = [reference]
    candidate_tokens = candidate

    chencherry = SmoothingFunction()

    # chencherry method 1 is real, method 2 is for machine translation, method5 gives best result.
    bleu_score = sentence_bleu(
        reference_tokens,
        candidate_tokens,
        weights=bleu_weights,
        smoothing_function=chencherry.method2)

    return bleu_score


def get_n_gram_weights(blue_index: int):
    weights = [
        (1. / 2., 1. / 2.),
        (1. / 3., 1. / 3., 1. / 3.),
        (1. / 4., 1. / 4., 1. / 4., 1. / 4.),
        (1. / 5., 1. / 5., 1. / 5., 1. / 5., 1. / .5),
    ]
    return weights[blue_index]


class BleuTest(unittest.TestCase):

    def test_bleu_score_calculation(self):
        reference_words = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
        predicted_words = ["the", "fast", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]

        bleu_score = calculate_bleu_score(reference_words, predicted_words, bleu_weights=get_n_gram_weights(2))
        print(f"BLEU Score: {bleu_score}")


if __name__ == "__main__":
    unittest.main()


