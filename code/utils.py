# Basic utilities
import os
import sys
import json
import logging
import string
import io

# Regex
import re

# Data Handling
import pandas as pd
import numpy as np

# Network and HTTP
import requests

# Mathematics and Random
import math
from random import randint, choice

# Serialization
import pickle

# Logging Configuration
# use: logging.info("Data has been backed up")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Tensorflow
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

# keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout, Bidirectional
from keras.optimizers import Adam, Adamax
from keras.preprocessing.text import Tokenizer

# NLP
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist
from nltk import ngrams

# For images
from PIL import Image
from IPython.display import display

import seaborn as sns
import matplotlib.pyplot as plt
