# Basic utilities
import os
import sys
import json
import logging
import argparse

# Regex
import re

# Data Handling
import panda as pd
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
