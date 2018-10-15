import abc
import re
import random
import numpy
import os
import pickle

from functools import reduce
from nltk.corpus import stopwords

from config import PathConfig, NetworkConfig
from utils.common import ProgressBar
from utils.db import Db

import warnings
warnings.filterwarnings(action="ignore", category=UserWarning, module="gensim")
from gensim.models import KeyedVectors


class Data(metaclass=abc.ABCMeta):
    __data__ = {"train": [[], []],
                "test": [[], []],
                "validation": [[], []],
                "max_seq_len": 0,
                "embeddings": numpy.zeros(0)}

    @staticmethod
    def __text__cleaning__(text):
        text = str(text).lower()
        text = re.sub(r"[!@#$%^&*()\-_+=\[{\]};:'\",<.>/?\\|~]", " ", text)
        text = re.sub(r"\d+", " ", text)
        text = re.sub(r"[\n\r\t]", " ", text)
        text = re.sub(r"\ss\s", " ", text)
        text = re.sub(r"\sre\s", " ", text)
        text = re.sub(r"\sve\s", " ", text)
        text = re.sub(r"\sd\s", " ", text)
        text = re.sub(r"\sll\s", " ", text)
        text = re.sub(r"\sm\s", " ", text)
        text = re.sub(r"\st\s", " ", text)

        return text.strip()

    @classmethod
    def __load__cache__(cls, cache_path):
        if not os.path.exists(cache_path):
            return False

        with open(cache_path, "rb") as cache_file:
            cls.__data__ = pickle.load(cache_file)

        return True

    @classmethod
    def __cache__(cls, cache_path):
        with open(cache_path, "wb") as cache_file:
            print("Caching into %s..." % cache_path)
            pickle.dump(cls.__data__, cache_file)
            print("Cached.\n")

    @classmethod
    def __generate__(cls):
        print("Loading word2vec from '%s'..." % PathConfig.W2V_FILE)
        word2vec = KeyedVectors.load_word2vec_format(PathConfig.W2V_FILE, binary=True)
        print("Word2vec has been loaded.\n")

        stops = set(stopwords.words("english"))
        vocabulary = dict()
        inverse_vocabulary = ["<UNKNOWN>"]
        max_seq_len = 0

        bar = ProgressBar(len(cls.__data__["train"][1]) +
                          len(cls.__data__["test"][1]) +
                          len(cls.__data__["validation"][1]),
                          "Generating numeric inputs...")
        progress = 0

        for set_name in "train", "test", "validation":
            raw_X = cls.__data__[set_name][0]
            Y = cls.__data__[set_name][1]

            X = []
            for pair in raw_X:
                numerated_pair = []
                for sentence in pair:
                    numerated_sentence = []
                    word_list = sentence.split(" ")
                    for word in word_list:
                        if word in stops and word not in word2vec:
                            continue

                        if word not in vocabulary:
                            vocabulary[word] = len(inverse_vocabulary)
                            numerated_sentence.append(vocabulary[word])
                            inverse_vocabulary.append(word)
                        else:
                            numerated_sentence.append(vocabulary[word])
                    numerated_pair.append(numerated_sentence)

                    if len(numerated_sentence) > max_seq_len:
                        max_seq_len = len(numerated_sentence)

                progress += 1
                bar.refresh(progress)

                X.append(numerated_pair)

            cls.__data__[set_name] = [X, Y]
            cls.__data__["max_seq_len"] = max_seq_len

        bar.finish("Numeric inputs has been generated.")

        bar = ProgressBar(len(vocabulary), "Generating word embeddings...")
        progress = 0

        embeddings = 1 * numpy.random.randn(len(vocabulary) + 1, NetworkConfig.EMBEDDING_DIM)
        for word, index in vocabulary.items():
            progress += 1
            bar.refresh(progress)
            if word in word2vec:
                embeddings[index] = word2vec[word]
        embeddings[0] = 0
        cls.__data__["embeddings"] = embeddings

        bar.finish("Word embeddings has been generated.")

    @abc.abstractmethod
    def load(self):
        return self

    @abc.abstractmethod
    def generate(self):
        return None

    @staticmethod
    def batches(data):
        X, Y = data[0], data[1]
        for i in range(0, len(X), NetworkConfig.BATCH_SIZE):
            remaining = len(X) - i
            if remaining < NetworkConfig.BATCH_SIZE:
                X_supplement = [[[0], [0]] for _ in range(NetworkConfig.BATCH_SIZE - remaining)]
                Y_supplement = [1.0 for _ in range(NetworkConfig.BATCH_SIZE - remaining)]
                batch = [X[i:] + X_supplement, Y[i:] + Y_supplement]
            else:
                batch = [X[i: i + NetworkConfig.BATCH_SIZE], Y[i: i + NetworkConfig.BATCH_SIZE]]
            yield batch


class SICK(Data):
    @classmethod
    def load(cls):
        if cls.__load__cache__(PathConfig.RAW_SICK_CACHE):
            print("Raw SICK data cache has been loaded from %s\n" % PathConfig.RAW_SICK_CACHE)
            return cls

        with open(PathConfig.SICK_FILE, "r", encoding="utf-8") as SICK_file:
            lines = SICK_file.readlines()
            bar = ProgressBar(len(lines) - 1, "Loading raw SICK data...")
            progress = 0

            for line in lines[1:]:
                elements = line.split("\t")
                sentence_A, sentence_B, relatedness_score = \
                    cls.__text__cleaning__(elements[1]), cls.__text__cleaning__(elements[2]), float(elements[4])
                SemEval_set = "validation" if elements[11].strip() == "TRIAL" else elements[11].strip().lower()

                cls.__data__[SemEval_set][0].append((sentence_A, sentence_B))
                cls.__data__[SemEval_set][1].append(relatedness_score / 5)

                progress += 1
                bar.refresh(progress)

        bar.finish("Raw SICK data has been loaded.")
        cls.__cache__(PathConfig.RAW_SICK_CACHE)

        return cls

    @classmethod
    def generate(cls):
        if cls.__load__cache__(PathConfig.SICK_CACHE):
            print("Numeric SICK inputs cache has been loaded from %s\n" % PathConfig.SICK_CACHE)
            return cls.__data__

        cls.__generate__()
        cls.__cache__(PathConfig.SICK_CACHE)

        return cls.__data__


class KES(Data):
    """This class loads our own private data. You may well ignore it."""

    @staticmethod
    def __self__accumulating__(n):
        return reduce(lambda x, y: x + y, range(1, n + 1))

    @staticmethod
    def __relatedness__score__(event_ids, sub_event_ids):
        assert isinstance(event_ids, tuple)
        assert isinstance(sub_event_ids, tuple)

        if event_ids[0] != event_ids[1]:
            return 0.2
        elif sub_event_ids[0] != sub_event_ids[1]:
            return 0.6
        else:
            return 1.0

    @classmethod
    def __fill__(cls, rows, row_count, set_name, progress_bar, progress):
        for i in range(row_count):
            for j in range(i + 1, row_count):
                cls.__data__[set_name][0].append((cls.__text__cleaning__(rows[i][2]),
                                                  cls.__text__cleaning__(rows[j][2])))
                cls.__data__[set_name][1].append(cls.__relatedness__score__((rows[i][0], rows[j][0]),
                                                                            (rows[i][1], rows[j][1])))

                progress += 1
                progress_bar.refresh(progress)

    @classmethod
    def load(cls):
        if cls.__load__cache__(PathConfig.RAW_KES_CACHE):
            print("Raw KES data cache has been loaded from %s\n" % PathConfig.RAW_KES_CACHE)
            return cls

        db = Db()
        rows = db.select(["new_eventid", "sub_eventid", "text_dbpedia"])
        random.shuffle(rows)
        row_count = len(rows)

        train_rows, test_rows, validation_rows = \
            rows[0: int(row_count * 0.45)], \
            rows[int(row_count * 0.45): int(row_count * 0.95)], \
            rows[int(row_count * 0.95):]

        bar = ProgressBar(cls.__self__accumulating__(len(train_rows) - 1) +
                          cls.__self__accumulating__(len(test_rows) - 1) +
                          cls.__self__accumulating__(len(validation_rows) - 1),
                          "Loading raw KES data...")
        progress = 0

        for rows_splitting, set_name in (train_rows, "train"), (test_rows, "test"), (validation_rows, "validation"):
            cls.__fill__(rows_splitting, len(rows_splitting), set_name, bar, progress)

        bar.finish("Raw KES data has been loaded.")
        cls.__cache__(PathConfig.RAW_KES_CACHE)

        return cls

    @classmethod
    def generate(cls):
        if cls.__load__cache__(PathConfig.KES_CACHE):
            print("Numeric KES inputs cache has been loaded from %s\n" % PathConfig.KES_CACHE)
            return cls.__data__

        cls.__generate__()
        cls.__cache__(PathConfig.KES_CACHE)

        return cls.__data__
