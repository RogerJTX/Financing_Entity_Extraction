from allennlp.data.dataset_readers import DatasetReader
from typing import Iterator, List, Dict
from allennlp.data.tokenizers import Token
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data import Instance



class NerReader(DatasetReader):
    """
    DatasetReader for PoS tagging data, one sentence per line, like

        [上海，国际，影视城]@@[B_LOC,B_ORG,I_ORG]
    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        
    def text_to_instance(self,tokens:List[str],tags:List[str] = None) -> Instance:
            
        tokens = [Token(word) for word in tokens]
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": sentence_field}

        if tags:
            label_field = SequenceLabelField(labels=tags, sequence_field = sentence_field)
            fields["labels"] = label_field

        return Instance(fields)
    
    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path, encoding='utf8') as f:
            for line in f:
                word_list, label_list = line.split("@@")
                word_list,label_list = eval(word_list), eval(label_list)
                yield self.text_to_instance(word_list,label_list)
                