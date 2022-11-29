import re
import os
import math
import requests
import networkx
import numpy as np
# from konlpy.tag import Okt
from konlpy.tag import Mecab

from rank_bm25 import BM25Okapi
import time

default_stopwords = []

class KoreanReviewSummarizerError(Exception):
    pass

class SentenceObj:

    def __init__(self, text, tokens=[], index=0):
        self.index = index
        self.text = text
        self.tokens = tokens

class Summarizer:
    def __init__(self, k=3
                     , useful_tags=None
                     , stopwords=None
                     , delimiter='\.|\\n|\.\\n|\!'
                     , spell_check=True
                     , return_all=False):
        # 시간 측정 시작
        # start = time.time()
        
        self.k = k
        # 품사 태그 설정
        if useful_tags==None:
            self.useful_tags=['Noun', 'Verb', 'Adjective', 'Determiner', 'Adverb', 'Conjunction', 'Josa', 'PreEomi', 'Eomi', 'Suffix', 'Alpha', 'Number']   # Okt 품사 태그
        else:
            self.useful_tags=useful_tags
        self.useful_tags=useful_tags
        # print(self.useful_tags)
        # 불용어 정하기
        if stopwords==None:
            cur_dir = os.path.dirname(__file__)
            f = open(cur_dir + "/default_korean_stopwords.txt", 'r')
            lines = f.readlines()
            for line in lines:
                line = line.replace('\n', '')
                default_stopwords.append(line)
            f.close()

            self.stopwords=default_stopwords
        else:
            self.stopwords=stopwords
        # print(self.stopwords)
        self.delimiter=delimiter
        self.spell_check=spell_check
        self.return_all=return_all
        # print("init time :", time.time() - start)

        # self.okt = Okt()      # okt
        self.mecab = Mecab()    # mecab
        if not isinstance(k, int):
            raise KoreanReviewSummarizerError('k must be int')
        # print("okt time :", time.time() - start)
        
        
    def summarize(self, reviews):
        # 시간 측정 시작
        start = time.time()

        if isinstance(reviews, list):
            reviews = ' '.join(reviews)
        self.splited_reviews = re.split(self.delimiter, reviews.strip())
        self.sentences = []
        self.sentence_index = 0

        # print("listify time :", time.time() - start)

        _agent = requests.Session()

        # print("session time :", time.time() - start)

        for one_sentence in self.splited_reviews:
            while len(one_sentence) and (one_sentence[-1] == '.' or one_sentence[-1] == ' '):
                one_sentence = one_sentence.strip(' ').strip('.')
            if not one_sentence:
                continue
            if self.spell_check:
                try:
                    base_url = 'https://m.search.naver.com/p/csearch/ocontent/spellchecker.nhn'
                    headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36'
                                    ,'referer': 'https://search.naver.com/'}
                    payload= {
                          '_callback': 'window.__jindo2_callback._spellingCheck_0'
                        , 'q': one_sentence
                    }
                    _checked = _agent.get(base_url, params=payload, headers=headers)
                    _checked = _checked.text[42:-2].split('\"html\":\"')[1].split('\"notag')[0]
                    _words = []
                    for word in _words.split('>'):
                        if not word.strip().startswith('<'):
                            _words.append(word.split('<')[0].strip())
                    one_sentence = ' '.join(_words)
                except:
                    pass

            # print("spell check time :", time.time() - start)
            
            tokens = []
            # print(one_sentence)
            # word_tag_pairs = self.okt.pos(one_sentence)               # okt
            # --- 여기서 처음 okt를 사용할 때 느려짐.. why..? --- 
            # print("okt word tag pair time :", time.time() - start)
            word_tag_pairs = self.mecab.pos(one_sentence)               # mecab
            # print(word_tag_pairs)
            # print("mecab word tag pair time :", time.time() - start)
            
            for word, tag in word_tag_pairs:
                if word in self.stopwords:
                    # print('stopwords :', word)
                    continue
                if tag not in self.useful_tags:
                    # print('useful_tags :', tag)
                    continue
                tokens.append("{}/{}".format(word, tag))
            if len(tokens) < 2:
                continue
            sentence = SentenceObj(one_sentence.strip(), tokens, self.sentence_index)
            self.sentences.append(sentence)
            self.sentence_index += 1

        # print("tokenize time :", time.time() - start)

        self.num_sentences = len(self.sentences)
        self.bm25 = BM25Okapi([sentenceObj.text for sentenceObj in self.sentences])
        for sentenceObj in self.sentences:
            sentenceObj.vector = self.bm25.get_scores(sentenceObj.text)
        
        # print("bm25 time :", time.time() - start)
            
        self.matrix = np.zeros((self.num_sentences, self.num_sentences))
        for sentence1 in self.sentences:
            for sentence2 in self.sentences:
                if sentence1 == sentence2:
                    self.matrix[sentence1.index, sentence2.index] = 1
                else:
                    self.matrix[sentence1.index, sentence2.index] = \
                    len(set(sentence1.tokens) & set(sentence2.tokens)) / \
                    (math.log(len(sentence1.tokens)) + math.log(len(sentence2.tokens)))
        
        # print("matrix time :", time.time() - start)

        self.graph = networkx.Graph()
        self.graph.add_nodes_from(self.sentences)
        for sentence1 in self.sentences:
            for sentence2 in self.sentences:
                weight = self.matrix[sentence1.index, sentence2.index]
                if weight:
                    self.graph.add_edge(sentence1, sentence2, weight=weight)

        # print("graph time :", time.time() - start)

        self.pagerank = networkx.pagerank(self.graph, weight='weight')
        self.result = sorted(self.pagerank, key=self.pagerank.get, reverse=True)

        # print("pagerank time :", time.time() - start)
        
        self.summaries = []
        if self.return_all:
            for i in range(len(self.result)):
                self.summaries.append(self.result[i].text)
                
            # return_all=True 시간 측정
            print("return_all(True) time :", time.time() - start)
            
            return self.summaries
        
        if self.k > len(self.result):
            for i in range(len(self.result)):
                self.summaries.append(self.result[i].text)
        else:
            for i in range(self.k):
                self.summaries.append(self.result[i].text)
            
        # return_all=False 시간 측정
        # print("return time :", time.time() - start)

        return self.summaries