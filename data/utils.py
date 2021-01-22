import jieba
import logging
import pymongo
import os
import random
from collections import defaultdict
import numpy as np
from logging import Filter
import sys
from allennlp.common.tqdm import Tqdm
logger = logging.getLogger(__name__)

def format_label(sentence_label:list):
    label_format = ["O"] * len(sentence_label)
    for i in range(len(sentence_label)):  
        label = sentence_label[i]
        if label == 'O':
            continue
        front_label = sentence_label[i-1]
        if label == front_label:
            label_format[i] = 'I_' + label
        else:
            label_format[i] = 'B_' + label
    return label_format

def get_word_entities(word_list,words_entities):
    '''
    将分词过的实体开始结束位置转化为字的开始结束位置
    开始结束跟标注平台一致，包含开始位置，不包含结束位置
    '''
    word_entities = []
    for entity,start,end in words_entities:
        entity_json = {}
        start = len("".join(word_list[:start]))
        end = len("".join(word_list[:end+1]))
        entity_json["start"] = start
        entity_json["end"] = end
        entity_json["entityType"] = entity
        entity_json["entity"] = "".join(word_list)[start:end]
        word_entities.append(entity_json)
    return word_entities



def gen_batch(size,X,Y):
    for i in range(0,len(X),size):
        batch_X, batch_Y = X[i:i+size], Y[i:i+size]
        for index in range(len(batch_X)):
            batch_X[index], batch_Y[index] = split_word_format(batch_X[index], batch_Y[index])
        yield batch_X, batch_Y


def split_word_format(sentence:str,labels:list):
    '''
    transfer sentence to split word list
    '''
    # 这里不能strip，语料中有空格字符
    # sentence = sentence.strip()
    sentence_list = list(jieba.cut(sentence,HMM=False))
    pointer = 0
    label_list = []
    for word in sentence_list:
        start = pointer
        end = pointer + len(word) + 1
        label = labels[start:end]
        word_label = ""
        if label[0].startswith("B") or label[0].startswith("I"):
            word_label = label[0]
        else:
            word_label = "O"
        label_list.append(word_label)
        pointer += len(word)
    if len(sentence_list) == len(label_list):
        return sentence_list,label_list
    else:
        logger.error("transform label error ")





def get_entities(seq, suffix=False):
    """Gets entities from sequence.

    Args:
        seq (list): sequence of labels.

    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).

    Example:
        >>> from seqeval.metrics.sequence_labeling import get_entities
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]
    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        if suffix:
            tag = chunk[-1]
            type_ = chunk.split('_')[0]
        else:
            tag = chunk[0]
            type_ = chunk.split('_')[-1]

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i-1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start


def classification_report(y_true, y_pred, digits=2, suffix=False):
    """Build a text report showing the main classification metrics.

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a classifier.
        digits : int. Number of digits for formatting output floating point values.

    Returns:
        report : string. Text summary of the precision, recall, F1 score for each class.

    Examples:
        >>> from seqeval.metrics import classification_report
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> print(classification_report(y_true, y_pred))
                     precision    recall  f1-score   support
        <BLANKLINE>
               MISC       0.00      0.00      0.00         1
                PER       1.00      1.00      1.00         1
        <BLANKLINE>
        avg / total       0.50      0.50      0.50         2
        <BLANKLINE>
    """
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    name_width = 0
    d1 = defaultdict(set)
    d2 = defaultdict(set)
    for e in true_entities:
        d1[e[0]].add((e[1], e[2]))
        name_width = max(name_width, len(e[0]))
    for e in pred_entities:
        d2[e[0]].add((e[1], e[2]))

    last_line_heading = 'avg / total'
    width = max(name_width, len(last_line_heading), digits)

    headers = ["precision", "recall", "f1-score", "support"]
    head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
    report = head_fmt.format(u'', *headers, width=width)
    report += u'\n\n'

    row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'

    ps, rs, f1s, s = [], [], [], []
    for type_name, true_entities in d1.items():
        pred_entities = d2[type_name]
        nb_correct = len(true_entities & pred_entities)
        nb_pred = len(pred_entities)
        nb_true = len(true_entities)

        p = 100 * nb_correct / nb_pred if nb_pred > 0 else 0
        r = 100 * nb_correct / nb_true if nb_true > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0

        report += row_fmt.format(*[type_name, p, r, f1, nb_true], width=width, digits=digits)

        ps.append(p)
        rs.append(r)
        f1s.append(f1)
        s.append(nb_true)

    report += u'\n'

    # compute averages
    report += row_fmt.format(last_line_heading,
                             np.average(ps, weights=s),
                             np.average(rs, weights=s),
                             np.average(f1s, weights=s),
                             np.sum(s),
                             width=width, digits=digits)

    return report


class FileFriendlyLogFilter(Filter):
    """
    TQDM and requests use carriage returns to get the training line to update for each batch
    without adding more lines to the terminal output.  Displaying those in a file won't work
    correctly, so we'll just make sure that each batch shows up on its one line.
    """

    def filter(self, record):
        if "\r" in record.msg:
            record.msg = record.msg.replace("\r", "")
            if not record.msg or record.msg[-1] != "\n":
                record.msg += "\n"
        return True


class WorkerLogFilter(Filter):
    def __init__(self, rank=-1):
        super().__init__()
        self._rank = rank

    def filter(self, record):
        if self._rank != -1:
            record.msg = f"Rank {self._rank} | {record.msg}"
        return True

def prepare_global_logging(
    serialization_dir: str, file_friendly_logging: bool, rank: int = 0, world_size: int = 1
) -> None:
    # If we don't have a terminal as stdout,
    # force tqdm to be nicer.
    if not sys.stdout.isatty():
        file_friendly_logging = True

    Tqdm.set_slower_interval(file_friendly_logging)

    # Handlers for stdout/err logging
    output_stream_log_handler = logging.StreamHandler(sys.stdout)
    error_stream_log_handler = logging.StreamHandler(sys.stderr)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")

    root_logger = logging.getLogger()

    # Remove the already set stream handler in root logger.
    # Not doing this will result in duplicate log messages
    # printed in the console
    if len(root_logger.handlers) > 0:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)

    # file handlers need to be handled for tqdm's \r char
    file_friendly_log_filter = FileFriendlyLogFilter()

    if os.environ.get("ALLENNLP_DEBUG"):
        LEVEL = logging.DEBUG
    else:
        LEVEL = logging.INFO

    if rank == 0:
        # stdout/stderr handlers are added only for the
        # master worker. This is to avoid cluttering the console
        # screen with too many log messages from all workers.
        output_stream_log_handler.setFormatter(formatter)
        error_stream_log_handler.setFormatter(formatter)

        output_stream_log_handler.setLevel(LEVEL)
        error_stream_log_handler.setLevel(logging.ERROR)

        if file_friendly_logging:
            output_stream_log_handler.addFilter(file_friendly_log_filter)
            error_stream_log_handler.addFilter(file_friendly_log_filter)

        root_logger.addHandler(output_stream_log_handler)
        root_logger.addHandler(error_stream_log_handler)

    root_logger.setLevel(LEVEL)


