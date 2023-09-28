from typing import Dict, List, Tuple, Optional

import torch

class BatchBuilder:
    # 1. Pass each sentence of the corpus through io parser.
    # 2. Take the input map add padding and special token and pass through get_batch_embedding_map
    # 3. Create src padding mask and future mask
    # 4. return the batch and mask map
    SOURCE_LANGUAGE_KEY = "src"
    TARGET_LANGUAGE_KEY = "tgt"

    @staticmethod
    def construct_batches(
            corpus: List[Dict[str, str]],
            batch_size: int,
            device: Optional[torch.device] = None,
    ) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, List[torch.Tensor]]]:
        """ Constructs batches given a corpus.

        :param corpus: The input corpus is a list of aligned source and target sequences, packed in a dictionary.
        :param batch_size: The number of sequences in a batch
        :param device: whether to move tensors to gpu
        :return: A tuple containing two dictionaries. The first represents the batches, the second the attention masks.
        """
        pad_token_id = vocab.token2index[vocab.PAD]
        batches: Dict[str, List] = {"src": [], "tgt": []}
        masks: Dict[str, List] = {"src": [], "tgt": []}
        for i in range(0, len(corpus), batch_size):
            src_batch = torch.IntTensor(
                vocab.batch_encode(
                    [pair[src_lang_key] for pair in corpus[i: i + batch_size]],
                    add_special_tokens=True,
                    padding=True,
                )
            )
            tgt_batch = torch.IntTensor(
                vocab.batch_encode(
                    [pair[tgt_lang_key] for pair in corpus[i: i + batch_size]],
                    add_special_tokens=True,
                    padding=True,
                )
            )

            src_padding_mask = src_batch != pad_token_id
            future_mask = construct_future_mask(tgt_batch.shape[-1])

            # Move tensors to gpu; if available
            if device is not None:
                src_batch = src_batch.to(device)  # type: ignore
                tgt_batch = tgt_batch.to(device)  # type: ignore
                src_padding_mask = src_padding_mask.to(device)
                future_mask = future_mask.to(device)
            batches["src"].append(src_batch)
            batches["tgt"].append(tgt_batch)
            masks["src"].append(src_padding_mask)
            masks["tgt"].append(future_mask)
        return batches, masks

    def construct_future_mask(seq_len: int):
        """
        Construct a binary mask that contains 1's for all valid connections and 0's for all outgoing future connections.
        This mask will be applied to the attention logits in decoder self-attention such that all logits with a 0 mask
        are set to -inf.

        :param seq_len: length of the input sequence
        :return: (seq_len,seq_len) mask
        """
        subsequent_mask = torch.triu(torch.full((seq_len, seq_len), 1), diagonal=1)
        return subsequent_mask == 0
