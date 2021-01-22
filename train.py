from data.utils import prepare_global_logging
from allennlp.common import Params
import json
from model.lstmCrf import LstmCrfTagger
import os
import torch
from data.reader import NerReader
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.common import Params
import torch.optim as optim
import logging
from allennlp.common.util import dump_metrics
from allennlp.models.archival import archive_model, load_archive

logger = logging.getLogger(__name__)

PRE_TRAINED_PATH = "/home/liangzhi/AI_services/supermind/autonlp/data/resources/"
CONFIG_NAME = "config.json"
MODEL_PATH = "output"
WEIGHTS_NAME = "best.th"
LEARNING_RATE = 0.01
EPOCH = 40
HIDDEN_DIM = 200
EMBEDDING_DIM = 300 # 加载的人民网预训练词向量维度
DROPOUT = 0.4
BATCH_SIZE = 64
SEED = 2020

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # if you are using multi-GPU.
torch.manual_seed(SEED)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def train():
 
    train_file = "data/train.txt"
    valid_file = "data/valid.txt"
    prepare_global_logging(MODEL_PATH, True)


    # read data into allennlp
    reader = NerReader()
    train_data = reader.read(train_file)
    valid_data = reader.read(valid_file)
    vocab = Vocabulary.from_instances(train_data + valid_data)
    params = {}
    params["embedding_dim"] = EMBEDDING_DIM
    params["pretrained_file"] = PRE_TRAINED_PATH + "sgns.renmin.word.gz"
    params["trainable"] = False
    params = Params(params=params)
    token_embedding = Embedding.from_params(vocab=vocab, params=params)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first = True, bidirectional = True))
    model = LstmCrfTagger(vocab = vocab, text_field_embedder = word_embeddings, encoder = lstm, dropout=DROPOUT)
    if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1
    logger.info("use cuda device {}".format(cuda_device))
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    iterator = BucketIterator(batch_size = BATCH_SIZE, sorting_keys = [("tokens","num_tokens")])
    iterator.index_with(vocab)
    trainer = Trainer(
                      model = model,
                      optimizer = optimizer,
                      iterator = iterator,              
                      train_dataset = train_data,
                      validation_dataset = valid_data,
                      serialization_dir = MODEL_PATH,
                      num_epochs = EPOCH,
                      cuda_device = cuda_device
                    )
    metrics = trainer.train()
    dump_metrics(os.path.join(MODEL_PATH, f"metrics.json"), metrics)
    logger.info("model training complete, save model starting")

    # save model to output
    vocab_dir = os.path.join(MODEL_PATH, "vocabulary")
    logger.info(f"writing the vocabulary to {vocab_dir}.")
    vocab.save_to_files(vocab_dir)
    logger.info("done creating vocab")
    archive_model(serialization_dir = MODEL_PATH, weights = WEIGHTS_NAME)


if __name__ == '__main__':
    train()
