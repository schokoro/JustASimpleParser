import pandas as pd
import json
from gc import collect
import re
from nltk.corpus import stopwords


def make_corps(jstrin):
    tokens = []
    for token in json.loads(jstrin):
        if token['lemma'] not in stopWords:
            lemma = re.sub(r'\W', '', token['lemma'])
            if lemma:
                tokens.append(lemma.lower())
    return ' '.join(tokens)


stopWords = set(stopwords.words())

train = pd.read_csv('../data/train_parsed.csv')#, usecols=['content'])

corps = pd.DataFrame(index = train.index)
corps['content'] = train.content.apply(lambda x: make_corps(x))

corps.to_csv('../data/train_corps.csv')
train.drop('content', axis=1,inplace=True)
train.to_pickle('../data/train.pickle', compression='xz')

del train, corps
collect()

test = pd.read_csv('../data/test_parsed.csv')
corps = pd.DataFrame(index = test.index)
corps['content'] = test.content.apply(lambda x: make_corps(x))
corps.to_csv('../data/test_corps.csv')
test.drop('content', axis=1,inplace=True)
test.to_pickle('../data/test.pickle', compression='xz')

