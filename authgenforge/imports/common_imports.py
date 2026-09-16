from __future__ import annotations

import csv
import gc
import glob
import io
import logging
import os
import random
import shutil
import yaml

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import cv2

cv2.setNumThreads(0)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from PIL import Image
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)