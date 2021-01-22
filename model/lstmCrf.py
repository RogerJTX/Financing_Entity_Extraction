import torch
import numpy as np
from typing import List, Dict,cast,Optional
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.models.crf_tagger import CrfTagger
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
import logging

logger = logging.getLogger(__name__)



class LstmCrfTagger(CrfTagger):
    
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder, 
                 dropout:Optional[float] = None       
                 ) -> None:
        super().__init__(vocab,text_field_embedder,encoder,calculate_span_f1=True,label_encoding="BIO",dropout=dropout) 

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        return super().forward(tokens,labels) 