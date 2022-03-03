import logging
import numpy as np
from demo_shap_adult import get_adult_dataset
from demo_shap_adult import get_examples


def test_read():
    df = get_adult_dataset()
    df.info()


def test_get_examples():
    prob = np.array([0.1, 0.1, 0.5, 0.9, 0.9])
    targ = np.array([0.0, 1.0, 0.0, 0.0, 1.0])
    e = get_examples(prob, targ)
    logging.info(e)
