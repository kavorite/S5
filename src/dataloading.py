import pickle
from functools import partial
import torch
from torch import nn
import torchtext

from pathlib import Path
from datasets import load_dataset, DatasetDict


# IMDB classification
def create_and_cache_imdb_dataset():
    path = Path('./S5/data/imdb')
    path.mkdir(parents=True, exist_ok=True)

    min_freq = 15
    l_max_raw = 2_048
    append_bos = False
    append_eos = True
    num_proc = 1

    def _do_tokenize(tokenizer, l_max, example):
        return {"tokens": tokenizer(example["text"])[:l_max]}

    def _numericalize(vocab, append_bos, append_eos, example):
        return {
            "input_ids": vocab(
                (["<bos>"] if append_bos else [])
                + example["tokens"]
                + (["<eos>"] if append_eos else [])
            )
        }

    # Load these datasets using the `dataset` library.
    # Reaches into the cache made by tfds.
    full_dataset = load_dataset('imdb')

    # Just convert a string to a list of chars
    tokenizer = list

    # Account for <bos> and <eos> tokens
    l_max = l_max_raw - int(append_bos) - int(append_eos)

    # Tokenize the data.
    _do_tokenize_closed = partial(_do_tokenize, tokenizer, l_max)
    dataset = full_dataset.map(
        _do_tokenize_closed,
        remove_columns=["text"],
        keep_in_memory=True,
        load_from_cache_file=False,
        num_proc=num_proc,
    )

    # Solve for the vocab.
    vocab = torchtext.vocab.build_vocab_from_iterator(
        dataset["train"]["tokens"],
        min_freq=min_freq,
        specials=(["<pad>", "<unk>"] + (["<bos>"] if append_bos else []) + (["<eos>"] if append_eos else [])
                  ),
    )
    vocab.set_default_index(vocab["<unk>"])

    # Numericalize the data.
    _numericalize_closed = partial(_numericalize, vocab, append_bos, append_eos)
    dataset_tokenized = dataset.map(
        _numericalize_closed,
        remove_columns=["tokens"],
        keep_in_memory=True,
        load_from_cache_file=False,
        num_proc=num_proc,
    )

    # Convert into torch.
    dataset_tokenized.set_format(type="torch", columns=["input_ids", "label"])

    trn_dset_obj = dataset_tokenized['train']
    trn_dset_obj.vocab = vocab
    with open('./S5/data/imdb/imdb_trn.p', 'wb') as f:
        pickle.dump(trn_dset_obj, f)

    tst_dset_obj = dataset_tokenized['test']
    tst_dset_obj.vocab = vocab
    with open('./S5/data/imdb/imdb_tst.p', 'wb') as f:
        pickle.dump(tst_dset_obj, f)

    return None


def make_imdb_data_loader(file_path, seed, batch_size=128, shuffle=True, drop_last=True):
    # Load the pickled dataset.
    with open(file_path, 'rb') as f:
        dset = pickle.load(f)

    # We need to define a custom collator that pads and returns the length.
    def _collate_batch(vocab, batch):
        xs, ys = zip(*[(data["input_ids"], data["label"]) for data in batch])
        lengths = torch.tensor([len(x) for x in xs])
        xs = nn.utils.rnn.pad_sequence(xs, padding_value=vocab["<pad>"], batch_first=True)
        ys = torch.tensor(ys)
        return xs, ys, lengths

    # Close over vocab for the collater to access later.
    collate_fn = partial(_collate_batch, dset.vocab)

    # Create a generator for seeding random number draws.
    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

    # Generate the dataloaders.
    return torch.utils.data.DataLoader(dataset=dset, collate_fn=collate_fn, batch_size=batch_size, shuffle=shuffle,
                                       drop_last=drop_last, generator=rng)


def create_imdb_classification_dataset(dir_name, bsz=50, seed=42):
    print("[*] Generating LRA-text (IMDB) Classification Dataset")
    SEQ_LENGTH, N_CLASSES, IN_DIM, TRAIN_SIZE = 2048, 2, 135, 25000

    trainloader = make_imdb_data_loader(dir_name + "/imdb/imdb_trn.p", seed=seed, batch_size=bsz)
    testloader = make_imdb_data_loader(dir_name + "/imdb/imdb_tst.p",
                                       seed=seed,
                                       batch_size=bsz,
                                       drop_last=False,
                                       shuffle=False
                                       )

    # Note we will use test set as validation set as done in LRA paper and S4 paper
    # see line 811 in https://github.com/HazyResearch/state-spaces/blob/main/src/dataloaders/datasets.py
    # for confirmation of this
    valloader = None
    return trainloader, valloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


Datasets = {
    "imdb-classification": create_imdb_classification_dataset
            }
