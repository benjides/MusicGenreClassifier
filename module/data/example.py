""" Example Loader Module """
import json
import numpy as np

root = 'datasets/train/'
separator = '---'

def get_example(row):
    """Gets the values for an example"""
    mbid = row[0]
    labels = row[2:].tolist()
    return get(mbid, labels)

def get(mbid, labels):
    """Gets the values for an example"""
    with open(root + mbid[:2] + '/' + mbid + '.json', 'r') as jsonfile:
        data = json.load(jsonfile)
        data = get_data(data)
        labels = get_labels(labels)
        return (data, labels)

def get_data(data):
    """Defines the data of the example

    Parameters
    ----------
        data: raw json from the training
    """
    return np.array(data["lowlevel"]["mfcc"]["mean"])

def get_labels(labels):
    """Defines the labels of the example as hierarchical dictionary

    Parameters
    ----------
        labels: string with the concatenated labels and sublabels
    """
    l = {}
    for label in labels:
        feed_label(label, l)
    return l

def feed_label(label, labels):
    """Sets the labels recursively

    Parameters
    ----------
        label: single label from the the groundtruth
        labels: dictionary with the keys
    """
    if not label:
        return
    chunks = label.split(separator)
    if chunks[0] not in labels.keys():
        labels[chunks[0]] = {}
    if len(chunks) == 1:
        return
    return feed_label(separator.join(chunks[1:]), labels[chunks[0]])
