from itertools import permutations
from server.config import config

END_OF_SENTENCE = config['end_of_sentence']

def synthesize(series, max_splits = 2):
    grouped = [ group_sentences(tokens, max_splits) for tokens in series ]
    expanded = []
    for group in grouped:
        for i, x in enumerate(permutations(group)):
            # we've already got the "real" data on our frame, don't bother including copies of the original in response
            if i != 0:
                expanded.append(flatten(x))

    return expanded


def find_split(i, available_splits, total_splits):
    return ((i + 1) * max((available_splits // total_splits), 1)) - 1

def group_sentences(tokens, max_splits = 2):
    av_splits = []
    for i in range(len(tokens)):
        if tokens[i] == END_OF_SENTENCE and i != 0 and i != (len(tokens) - 1):
            av_splits.append(i)

    collection = []
    total_splits = min(max_splits, len(av_splits))
    calc_split = lambda x: find_split(x, len(av_splits), total_splits)

    if total_splits == 0:
        collection.append(tokens)
    else:
        for i in range(total_splits + 1): # Total groups is 1 more than total splits
            if i == 0:
                collection.append(tokens[:av_splits[calc_split(i)]])
            elif i == total_splits:
                collection.append(tokens[av_splits[calc_split(i - 1)]:])
            else:
                collection.append(tokens[av_splits[calc_split(i - 1)]: av_splits[calc_split(i)]])

    return collection

def flatten(arr):
    return [ x for sub in arr for x in sub ]