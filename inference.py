import torch
from torch import tensor
from torchtext.vocab.vocab import Vocab
from torch.utils.data.dataloader import DataLoader
from transformer.encoder_decoder import EncoderDecoder

import transformer.loaders as loaders
from transformer.builder import make_model
from transformer.masks import subsequent_mask

def greedy_decode(
        model: EncoderDecoder, 
        src: tensor,
        src_mask: tensor,
        max_len: int,
        start_symbol: int
):
    """
    Args:
        src: (N, max_seq)
        src_mask: (N, 1, max_seq)
    """
    memory = model.encode(src, src_mask)
    batch_sz = src.shape[0]
    # The first input passed as tgt to the decoder is the start token
    ys = torch.zeros(batch_sz, 1).fill_(start_symbol).type_as(src.data)

    for i in range(max_len - 1):

        out = model.decode(
            memory=memory, 
            src_mask=src_mask,
            tgt=ys,
            tgt_mask=subsequent_mask(ys.size(1)).type_as(src.data)
        )

        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(batch_sz, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys


def check_outputs(
    valid_dataloader: DataLoader,
    model: EncoderDecoder,
    vocab_src: Vocab,
    vocab_tgt: Vocab,
    n_examples: int=15,
    pad_idx: int=2,
    eos_string: str="</s>",
):
    results = [()] * n_examples
    for idx in range(n_examples):
        print("\nExample %d ========\n" % idx)
        b = next(iter(valid_dataloader))
        rb = loaders.Batch(src=b[0], tgt=b[1], pad=pad_idx)
        
        greedy_decode(model, rb.src, rb.src_mask, max_len=64, start_symbol=0)[0]

        # ['<s>', 'Ein', 'älterer', 'Mann' 'sitzt', 'im' 'Freien', 'vor', 'einem', 'großen' 'Banner' , 'mit', 'der', 'Aufschrift',
        src_tokens = [
            vocab_src.get_itos()[x] for x in rb.src[0] if x != pad_idx
        ]
        ## tgt_tokens = ['An', 'older', 'man', 'is', 'sitting', 'outside', 'on', 'a', 'bench', 'in', 'front', 'a', 'large', •••J]
        tgt_tokens = [
            vocab_tgt.get_itos()[x] for x in rb.tgt[0] if x != pad_idx
        ]

        print(
            "Source Text (Input)        : "
            + " ".join(src_tokens).replace("\n", "")
        )
        print(
            "Target Text (Ground Truth) : "
            + " ".join(tgt_tokens).replace("\n", "")
        )
        model_out = greedy_decode(model, rb.src, rb.src_mask, 72, 0)[0]
        model_txt = (
            " ".join(
                [vocab_tgt.get_itos()[x] for x in model_out if x != pad_idx]
            ).split(eos_string, 1)[0]
            + eos_string
        )
        print("Model Output               : " + model_txt.replace("\n", ""))
        results[idx] = (rb, src_tokens, tgt_tokens, model_out, model_txt)
    return results


def run_model_example(n_examples: int=5):

    print("Preparing Data ...")
    spacy_de, spacy_en = loaders.load_tokenizers()
    vocab_src, vocab_tgt = loaders.load_vocab(spacy_de, spacy_en)

    _, valid_dataloader = loaders.create_dataloaders(
        torch.device("cpu"),
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=2,
        is_distributed=False,
    )

    print("Loading Trained Model ...")

    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(
        torch.load("data/multi30k_model_final.pt", map_location=torch.device("cpu"))
    )

    print("Checking Model Outputs:")
    example_data = check_outputs(
        valid_dataloader, model, vocab_src, vocab_tgt, n_examples=n_examples
    )
    return model, example_data


if __name__ == "__main__":
    run_model_example()
