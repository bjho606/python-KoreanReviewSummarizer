import re
import math
import requests
import networkx
import numpy as np
from konlpy.tag import Okt
from rank_bm25 import BM25Okapi

stopwords = ['예뻐', '예뻐요', '예쁩니다', '예쁘고', '예쁘구', '이뻐', '이뻐요', '이쁩니다', '이쁘고','이쁘구', '좋아요', '좋습니다',
'괜찮아요', '만족해요', '만족', '배송', '빠른', '빨리', '빨랐어요', '빨라', '택배', '감사합니다', '고맙습니다', '많이 파세요', '최고',
'쿠폰', '할인', '같아요', '무난', '선물', '존예', '맘에 들어요', '맘에 듭니다']

class KoreanReviewSummarizerError(Exception):
    pass

class SentenceObj:

    def __init__(self, text, tokens=[], index=0):
        self.index = index
        self.text = text
        self.tokens = tokens

class Summarizer:

    def __init__(self, k=5
                     , useful_tags=['Noun', 'Verb', 'Adjective', 'Determiner', 'Adverb', 'Conjunction', 'Josa', 'PreEomi', 'Eomi', 'Suffix', 'Alpha', 'Number']
                     , stopwords=stopwords
                     , spell_check=True
                     , return_all=False):
        self.k = k
        self.useful_tags=useful_tags
        self.stopwords=stopwords
        self.spell_check=spell_check
        self.return_all=return_all
        self.okt = Okt()
        if not isinstance(k, int):
            raise KoreanTextRank4ReviewError('k must be int')
        
        
    def summarize(self, reviews):
        if isinstance(reviews, list):
            reviews = ' '.join(reviews)
        self.splited_reviews = re.split('\.|\\n|\.\\n|\!', reviews.strip())
        self.sentences = []
        self.sentence_index = 0
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
                    _agent = requests.Session()
                    _checked = _agent.get(base_url, params=payload, headers=headers
                    _checked = _checked.text[42:-2].split('\"html\":\"')[1].split('\"notag')[0]
                    _words = []
                    for word in words.split('>'):
                        if not word.strip().startswith('<'):
                            _words.append(word.split('<')[0].strip())
                    one_sentence = ' '.join(_words)
                except:
                    pass
            tokens = []
            word_tag_pairs = self.okt.pos(one_sentence)
            for word, tag in word_tag_pairs:
                if word in self.stopwords:
                    continue
                if tag not in self.useful_tags:
                    continue
                tokens.append("{}/{}".format(word, tag))
            if len(tokens) < 2:
                continue
            sentence = SentenceObj(one_sentence.strip(), tokens, self.sentence_index)
            self.sentences.append(sentence)
            self.sentence_index += 1

        self.num_sentences = len(self.sentences)
        self.bm25 = BM25Okapi([sentenceObj.text for sentenceObj in self.sentences])
        for sentenceObj in self.sentences:
            sentenceObj.vector = self.bm25.get_scores(sentenceObj.text)
            
        self.matrix = np.zeros((self.num_sentences, self.num_sentences))
        for sentence1 in self.sentences:
            for sentence2 in self.sentences:
                if sentence1 == sentence2:
                    self.matrix[sentence1.index, sentence2.index] = 1
                else:
                    self.matrix[sentence1.index, sentence2.index] = \
                    len(set(sentence1.tokens) & set(sentence2.tokens)) / \
                    (math.log(len(sentence1.tokens)) + math.log(len(sentence2.tokens)))
        
        self.graph = networkx.Graph()
        self.graph.add_nodes_from(self.sentences)
        for sentence1 in self.sentences:
            for sentence2 in self.sentences:
                weight = self.matrix[sentence1.index, sentence2.index]
                if weight:
                    self.graph.add_edge(sentence1, sentence2, weight=weight)
        self.pagerank = networkx.pagerank(self.graph, weight='weight')
        self.result = sorted(self.pagerank, key=self.pagerank.get, reverse=True)
        
        self.summaries = []
        if self.return_all:
            for i in range(len(self.result)):
                self.summaries.append(self.result[i].text)
                
            return self.summaries
            
        for i in range(self.k):
            self.summaries.append(self.result[i].text)
            
        return self.summaries