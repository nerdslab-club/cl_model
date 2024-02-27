import unittest

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu

from cl_pretrainer.batch_builder import BatchBuilder
from cl_pretrainer.pre_trainer_utils import PreTrainerUtils


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
    total_score = 0
    total_count = 0

    for index, (reference_tokens, candidate_tokens) in enumerate(zip(reference, candidate)):
        bleu_score = calculate_sentence_bleu_score(
            reference_tokens,
            candidate_tokens,
            bleu_weights=bleu_weights)
        total_score += bleu_score
        total_count += 1

    corpus_bleu_score = round(total_score/total_count, 2)
    return corpus_bleu_score


def get_n_gram_weights(blue_index: int):
    blue_index = blue_index - 2
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
        self.assertAlmostEqual(0.8314794192830981, bleu_score)

    def test_sentence_bleu_score_calculation_using_io_parser(self):
        # Creating the vocabulary corpus
        sentences = [
            "The quick brown fox jumps over the lazy dog in the meadow",
            "Adding 3 plus 2 equals ##addition(3,2)",
            "Each children will receive ##division(9,3) candies",
            "The result of subtracting 1 from 5 is ##subtraction(5,1)",
        ]

        batches, masks = BatchBuilder.construct_batches_for_cl_pre_trainer(
            sentences,
            batch_size=2,
            max_decoder_sequence_length=16,
            is_generative_training=False,
        )

        reference_sentences = PreTrainerUtils.extract_tokens(batches[BatchBuilder.ENCODER_IO_PARSER_OUTPUT_KEY][0])
        predicted_sentence = PreTrainerUtils.extract_tokens(batches[BatchBuilder.ENCODER_IO_PARSER_OUTPUT_KEY][0])

        bleu_score = calculate_corpus_bleu_score(
            reference_sentences,
            predicted_sentence,
            bleu_weights=get_n_gram_weights(2))
        print(f"BLEU Score of the corpus: {bleu_score}")
        self.assertAlmostEqual(bleu_score, 1)

    def test_corpus_bleu_score_calculation(self):
        reference_corpus = [
            ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"],
            ["jumps", "over", "the", "lazy", "dog"],
        ]
        predicted_corpus = [
            ["the", "fast", "brown", "fox", "jumps", "over", "the", "lazy", "dog"],
            ["jumps", "over", "the", "lazy", "dog"],
        ]

        bleu_score = calculate_corpus_bleu_score(reference_corpus, predicted_corpus, bleu_weights=get_n_gram_weights(2))
        print(f"BLEU Score of the corpus: {bleu_score}")
        self.assertAlmostEqual(0.92, bleu_score)


if __name__ == "__main__":
    unittest.main()


