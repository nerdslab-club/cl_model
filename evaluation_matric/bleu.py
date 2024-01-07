import unittest

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu


def calculate_sentence_bleu_score(reference: list[str], candidate: list[str], bleu_weights: tuple):
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


def calculate_corpus_bleu_score(reference: list[list[str]], candidate: list[list[str]], bleu_weights: tuple):
    """Calculate BLEU score between reference and candidate sentences.

    :param reference: List of List of reference words (list of strings)
    :param candidate: List of List of candidate words (list of strings)
    :param bleu_weights: How manu n-grams to consider.
    :return: BLEU score (float)
    """
    # Convert the reference and candidate sentences into a list of lists of tokens
    reference_tokens = reference
    candidate_tokens = candidate

    chencherry = SmoothingFunction()

    # chencherry method 1 is real, method 2 is for machine translation, method5 gives best result.
    bleu_score = corpus_bleu(
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

    def test_sentence_bleu_score_calculation(self):
        reference_sentence = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
        predicted_sentence = ["the", "fast", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]

        bleu_score = calculate_sentence_bleu_score(reference_sentence, predicted_sentence, bleu_weights=get_n_gram_weights(2))
        print(f"BLEU Score of the sentence: {bleu_score}")

    def test_corpus_bleu_score_calculation(self):
        reference_corpus = [
            [["the", "quick", "brown", "fox"], ["the", "fast", "brown", "fox"]],
            [["jumps", "over", "the", "lazy", "dog"]],
        ]
        predicted_corpus = [
            ["the", "fast", "brown", "fox"],
            ["jumps", "over", "the", "lazy", "dog"],
        ]

        bleu_score = calculate_corpus_bleu_score(reference_corpus, predicted_corpus, bleu_weights=get_n_gram_weights(2))
        print(f"BLEU Score of the corpus: {bleu_score}")


if __name__ == "__main__":
    unittest.main()


