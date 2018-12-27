import json
import pandas as pd
import re
from dateutil import parser as dateparser
from janome.tokenizer import Tokenizer as JTokenizer
import unicodedata
import spacy
import csv
from os import path
from gc import collect
from bs4 import BeautifulSoup
import pycld2 as cld2
from chop.mmseg import Tokenizer as MMSEGTokenizer
from pymystem3 import Mystem
from polyglot.text import Text
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
import pdb

class JMParser:
    def __init__(self):
        """
        Инициализация лемматайзеров и регулярок
        """
        self.spacy_langs = ('en', 'de', 'es', 'pt', 'fr', 'it', 'nl')
        self.mystem = Mystem()
        print('MyStem lemmatizer is created')
        self.nlp = {lang: spacy.load(lang) for lang in self.spacy_langs}
        print('Spacy lemmatizers are created')
        self.mt = MMSEGTokenizer()
        print('MMSEGTokenizer is created')
        self.jtokenizer = JTokenizer()
        print('JTokenizer is created')
        self.remove_links = re.compile(r'(http[s]?://([a-zA-Z0-9_]+\.)+[a-zA-Z]{2,6}/\S*)')
        self.remove_space = re.compile(r'[\s]{2,}|(\u200b)')
        self.remove_puncts = re.compile(r"""[\<\>\(\)\{\[\]\}\.\?\!\;\:]+""")
        self.re_uni = re.compile(r'\W')
        self.separation = re.compile(r'(<p^[>]*>)|(<div^[>]*>)|(</p>)|(</div>)|(\\n)|(\\r)')
        self.sep_header = re.compile(r'(</h\d?>)')
        self.remove_quotes = re.compile(r'(\u2018)|(\u2019)')
        self.bad_pattern = re.compile(r'(\x00)|(\x01)|(\x02)|(\x03)|(\x04)|(\x05)|\
        (\x06)|(\x07)|(\x08)|(\x0b)|(\x0e)|(\x0f)|(\x10)|(\x11)|(\x12)|(\x13)|\
        (\x14)|(\x15)|(\x16)|(\x17)|(\x18)|(\x19)|(\x1a)|(\x1b)|(\x1c)|(\x1d)|\
        (\x1e)|(\x1f)|(\x7f)|(\x80)|(\x81)|(\x82)|(\x83)|(\x84)|(\x85)|(\x86)|\
        (\x87)|(\x88)|(\x89)|(\x8a)|(\x8b)|(\x8c)|(\x8d)|(\x8e)|(\x8f)|(\x90)|\
        (\x91)|(\x92)|(\x93)|(\x94)|(\x95)|(\x96)|(\x97)|(\x98)|(\x99)|(\x9a)|\
        (\x9b)|(\x9c)|(\x9d)|(\x9e)|(\x9f)')
        self.remove_smiles = re.compile(r'[@]|[©-®]|[‼-㊙]|[🀄-🛅]', re.UNICODE)
        self.pos = ('PROPN', 'VERB', 'PART', 'NOUN', 'ADV',
                    'ADJ', 'CCONJ', 'PRON', 'INTJ', 'NUM', 'PART')
        print('Ready to parsing...')


    def parse(self, args):
        """
        Парсинг одной статьи в json-формате.
        :param args: кортеж из трёх значений:
            - id - произвольное число типа int
            - article - запись в формате json
            - target - таргет он и есть таргет
        :return: dictionary
        """
        id, article, target = args
        self.parser = ''
        article = json.loads(article)
        content, tags = self.get_jcontent(article['content'])
        if not content:
            content = 'not content'
        author = re.findall(r'(@.*$)', article['author']['url'])[0]
        dttm = dateparser.parse(article['published']['$date'])
        day_of_week = dttm.weekday()
        time = str(dttm.time())
        minutes = dttm.minute + 60 * dttm.hour
        # hours = dttm.hour
        self.url = article['url']
        date = str(dttm.date())
        if id % 1000 == 0:
            print(f'Удалено {collect()} объектов')

        return {
            'id': id,
            'author': author,
            'date': date,
            'time': time,
            'day_of_week': day_of_week,
            'minutes': minutes,
            'length': self.length,
            'lang': self.lang,
            'content': content,
            'tags': tags,
            'tittle': article['title'],
            'url': self.url,
            'parser': self.parser,
            'target': target
        }

    def get_jcontent(self, content):
        """

        :param content: 'content' из json
        :return: корпус статьи список тегов в формате json
        """
        corps = []
        content = unicodedata.normalize("NFKD", content)
        content = self.separation.sub(' \g<1> ', content)
        content = self.sep_header.sub('\g<1>. ', content)
        content = self.bad_pattern.sub(' ', content)
        content = self.remove_smiles.sub(' ', content)
        content = self.remove_quotes.sub("'", content)
        soup = BeautifulSoup(content, 'lxml')
        tags = self.get_tags(soup)
        content_blocks = soup.findAll('div', 'section-content')
        self.lang = 'un'
        for i, block in enumerate(content_blocks):
            try:
                block = self.remove_claps(block)
                content = BeautifulSoup(str(block), 'lxml').get_text()
                content = self.remove_links.sub(' ', content)
                content = self.remove_space.sub(' ', content)
                if i == 0 or self.lang == 'un':
                    self.gues_lang(content)
                corps += self.make_corps(content)
            except RuntimeError:
                pdb.set_trace()
                print('FUCK!')
                print(self.url)
        self.length = len(corps)
        return json.dumps(corps), json.dumps(tags)

    def remove_claps(self, block):
        divs = block.findAll('p', text=re.compile(r'(clap)')) # text=re.compile(r'(One clap)')
        if divs:
            clap_block = divs[0].find_parent()
            clap_block.contents = []
        return block

    def gues_lang(self, content):
        content = self.re_uni.sub(' ', content)
        self.lang = cld2.detect(content)[2][0][1]
        if self.lang == 'un':
            self.lang = Text(content).language.code
        i = 0
        l = int(len(content) / 10)
        while (self.lang == 'un') and (i <= 5):
            self.lang = cld2.detect(content[i * l: (i + 1) * l])[2][0][1]
            i += 1

    def get_tags(self, soup):
        tags = []
        try:
            for a in soup.find('ul', 'tags--postTags').findAll('a'):
                tags.append(a.get_text())
        except:
            pass
        return tags

    def make_corps(self, content):
        if self.lang in self.spacy_langs:
            corps = self.spacy_lemmatizer(content)
        elif self.lang == 'zh-Hant' or self.lang == 'zh_Hant' or self.lang == 'zh':
            if self.lang == 'zh-Hant':
                self.lang = 'zh_Hant'
            corps = self.chop_lemmatizer(content)
        elif self.lang == 'ru':
            corps = self.mystem_lemmatizer(content)
        elif self.lang == 'ja':
            corps = self.jap_lemmatizer(content)
        else:
            corps = self.generic_lemmatizer(content)
        return corps

    def spacy_lemmatizer(self, content):
        length = 0
        corps = []
        try:
            pre_tokens = self.nlp[self.lang](content)
        except (MemoryError, RuntimeError):
            self.lang = 'un'
            return corps
        for token in pre_tokens:
            if token.lemma_ == '-PRON-':
                corps.append({
                    'lexema': token.text,
                    'lemma': token.text,
                    'pos': token.pos_
                })
                length += 1
            elif token.pos_ in self.pos:
                corps.append({
                    'lexema': token.text,
                    'lemma': token.lemma_,
                    'pos': token.pos_
                })
                length += 1
        self.parser = 'spacy'
        return corps

    def mystem_lemmatizer(self, content):
        corps = []
        pos_pattern = re.compile(r'^\w+')
        text_pattern = re.compile(r'\w+')
        lemmas = self.mystem.analyze(content.lower())
        for analys in lemmas:
            lemma = analys.get('analysis')
            if lemma:
                corps.append({
                    'lexema': analys['text'],
                    'lemma': lemma[0]['lex'],
                    'pos': pos_pattern.match(lemma[0]['gr'].split(',')[0])[0]
                })
            elif text_pattern.match(analys.get('text')):
                token = text_pattern.match(analys.get('text'))[0]
                corps.append({
                    'lexema': token,
                    'lemma': token,
                    'pos': 'notrus'
                })
        self.parser = 'mystem'
        return corps

    def chop_lemmatizer(self, content):
        corps = []

        tokens = self.mt.cut(content)
        try:
            for i, token in enumerate(tokens):
                corps.append({
                    'lexema': token,
                    'lemma': token,
                    'pos': '词性'
                })
        except:
            corps = self.generic_lemmatizer(content)
        self.parser = 'chop'
        return corps

    def jap_lemmatizer(self, content):
        corps = []

        for token in self.jtokenizer.tokenize(content):
            lexema = token.base_form
            if re.match(r'\w+', lexema):
                pos = token.part_of_speech.split(',')[0]
                corps.append({
                'lexema': lexema,
                'lemma': lexema,
                'pos': pos
                })
        self.parser = 'janome'
        return corps

    def generic_lemmatizer(self, content):
        corps = []
        try:
            text = Text(content)
            corps = [{'lexema': token, 'lemma': token, 'pos': 'token'} for token in text.words if
                     not re.match(r'\W', token)]
            self.parser = 'polyglot'
        except:
            re_tokens = re.findall(r'(\w{2,})', content.lower())
            for token in re_tokens:
                corps.append({
                    'lexema': token,
                    'lemma': token,
                    'pos': 'token'
                })
            self.parser = 're'
        return corps


def parse(file_in='../data/test.json', file_out=None, target=None):
        if not file_out:
            path_in, name_in = path.split(file_in)
            file_out = path.join(path_in, path.splitext(name_in)[0] + '_parsed.csv')
        jmparser = JMParser()
        with open(file_in, 'r') as file_obj:
            record = file_obj.readlines()
        total = len(record)
        ids = list(range(total))
        if target:
            target = list(pd.read_csv(target)['log_recommends'])
        else:
            target = [0] * total
        args = zip(ids, record, target)
        with open(file_out, 'w') as out_obj:
            writer = csv.writer(out_obj)
            writer.writerow(['id', 'author', 'date', 'time', 'day_of_week',
                             'minutes', 'length', 'lang', 'content',
                             'tags', 'tittle', 'url', 'parser', 'target'])
        with open(file_out, 'a') as out_obj:
            writer = csv.writer(out_obj)
            for arg in tqdm(args, total=total):
                #if arg[0] < 21709:
                #    pass
                parsed = jmparser.parse(arg)
                csv_string = [
                    parsed['id'],
                    parsed['author'],
                    parsed['date'],
                    parsed['time'],
                    parsed['day_of_week'],
                    parsed['minutes'],
                    parsed['length'],
                    parsed['lang'],
                    parsed['content'],
                    parsed['tags'],
                    parsed['tittle'],
                    parsed['url'],
                    parsed['parser'],
                    parsed['target']
                ]
                writer.writerow(csv_string)
        collect()


if __name__ == '__main__':
    parse(file_in='../data/train.json',
                  file_out='../data/train_parsed.csv',
                  target='../data/train_log1p_recommends.csv')
    print('train  готов')
    parse(data='../data/test.json')
    print('test  готов')
    print('OK')
