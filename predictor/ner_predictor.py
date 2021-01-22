from data.reader import NerReader
from data.utils import get_entities,get_word_entities
from allennlp.data import Instance
import torch
import numpy as np
from allennlp.common.util import JsonDict
from typing import List, Dict
import jieba

class NerPredictor():
    def __init__(self, model):
        
        self.model = model
        self._dataset_reader = NerReader()

    def predict(self, doc: Dict) -> JsonDict:
        return self.predict_json(doc)

    def predict_json(self,doc:Dict):
        instance = self._json_to_instance(doc)
        return self.predict_instance(instance)

    def _json_to_instance(self, doc: JsonDict) -> Instance:

        sentence = doc['origin_content'][list(doc['origin_content'].keys())[0]]
        sentence_tokens = list(jieba.cut(sentence,HMM=False))
        return self._dataset_reader.text_to_instance(sentence_tokens)

    def predict_instance(self,instance:Instance) -> str:
        '''
        接收Instance，返回解析好的实体列表
        '''
        with torch.no_grad():
            instance.index_fields(self.model.vocab)
            tokens_list = instance.get("tokens")._indexed_tokens["tokens"]
            cuda_device = self.model._get_prediction_device()
            sentence_tensor = torch.LongTensor([tokens_list])
            if cuda_device != -1:
                sentence_tensor = sentence_tensor.cuda(cuda_device)
            tokens = {"tokens":sentence_tensor}
            tag_ids = self.model.eval().forward(tokens)["tags"][0]
            sentence_labels = [self.model.vocab.get_token_from_index(i, 'labels') for i in tag_ids]
            sentence_tokens = instance.get("tokens").tokens
            word_list = [str(token) for token in sentence_tokens]
            words_entities = get_entities(sentence_labels)
            entities = get_word_entities(word_list,words_entities)
            return entities




