class PathConfig:
    ROOT_DIR = "C:/Users/Weaine/PycharmProjects/siamese_lstm/"
    DATA_DIR = ROOT_DIR + "data/"
    CACHE_DIR = ROOT_DIR + "cache/"
    MODEL_PATH = ROOT_DIR + "model/"

    SICK_FILE = DATA_DIR + "SICK.txt"
    W2V_FILE = DATA_DIR + "GoogleNews-vectors-negative300.bin.gz"

    RAW_SICK_CACHE = CACHE_DIR + "SICK.raw.cache"
    RAW_KES_CACHE = CACHE_DIR + "KES.raw.cache"
    SICK_CACHE = CACHE_DIR + "SICK.cache"
    KES_CACHE = CACHE_DIR + "KES.cache"


class DbConfig:
    HOST = "10.10.7.251"
    PORT = 5432
    USER = "postgres"
    PASS = "postgres"
    DB = "lvshangwen"
    TABLE = "subtopicreferences_final"


class NetworkConfig:
    EMBEDDING_DIM = 300
    HIDDEN_SIZE = 50
    BATCH_SIZE = 64
    EPOCH = 5
