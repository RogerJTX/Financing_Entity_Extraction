from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
import torch
import logging
from model.lstmCrf import LstmCrfTagger
from predictor.ner_predictor import NerPredictor
from allennlp.common import Params

logger = logging.getLogger(__name__)

PRE_TRAINED_PATH = "resources/"
EMBEDDING_DIM = 300
HIDDEN_DIM = 200
DROPOUT = 0
SEED = 2020

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # if you are using multi-GPU.
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # Load model from output
    vocab = Vocabulary.from_files("output/vocabulary")
    params = {}
    params["embedding_dim"] = EMBEDDING_DIM
    params["pretrained_file"] = PRE_TRAINED_PATH + "sgns.renmin.word.gz"
    params["trainable"] = False
    params = Params(params=params)
    token_embedding = Embedding.from_params(vocab=vocab, params=params)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first = True, bidirectional = True))
    model = LstmCrfTagger(vocab = vocab, text_field_embedder = word_embeddings, encoder = lstm, dropout=DROPOUT)
    with open("output/best.th","rb") as f:
        model.load_state_dict(torch.load(f))

    if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1

    predictor = NerPredictor(model)
    querybody = {
        "origin_content":{
            "content":"近日获悉，国内全景硬件及方案厂商圆周率科技近期获得华润创新基金、力合创投新一轮数千万元融资。资金将主要用于团队建设、行业方案研发及产品生态建设。"
        }
    }
    while True:
        content = str(input())
        querybody = eval(content)
        pred_label = predictor.predict(querybody)
        res = {"label":pred_label}
        print(str(res))







