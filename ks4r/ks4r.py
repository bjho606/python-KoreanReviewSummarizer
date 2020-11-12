import re
import math
import requests
import networkx
import numpy as np
from konlpy.tag import Okt
from rank_bm25 import BM25Okapi

default_stopwords = ['좋아요', '너무', '신발', '배송', '좀', '딱', '것', '주문', '마음', '때', '생각', '신어', '신고', '들어요', '상품', '좋습니다', '더', '맘', '좋은', '좋네요', '이뻐요', '정말', '아주', '거', '예뻐요'
            , '이쁘고', '감사합니다', '신', '평소', '좋고', '느낌', '이쁘네요', '색상', '완전', '제', '예쁘고', '고민', '그래도', '근데', '여름', '저', '대비', '신으면', '원래', '보고', '받았습니다', '빠른', '발도', '이쁩니다',
            '듭니', '실물', '깔끔하고', '또', '아들', '하세요', '신을게요', '분', '버스', '신는', '매우', '괜찮아요', '사서', '하게', '자주', '그리고', '하지만', '걱정', '예쁘네요', '네', '매장', '사람', '일단', '블랙', '흰색', '다만', '이번'
            , '다른', '색깔', '아이', '커플', '같은', '할인', '있어', '하고', '될', '없이', '이뻐서', '바로', '샀습니다', '보면', '예쁩니다', '전부', '상태', '쏙', '개', '파세요', '쿠폰', '좋아하네요', '스타일', '기분', '사세요', '빨리'
            , '예뻐서', '믿고', '훨씬', '좋아서', '때문', '포장', '꼭', '있을', '괜찮네요', '포인트', '살', '들어', '있네요', '굳', '받았어요', '반품', '뭐', '개인', '사고', '좋은데', '이쁘게', '괜찮은', '색도', '모두', '아직', '효과', '친구'
            , '엄마', '샀네요', '아직', '그대로', '다시', '만족해요', '좋아', '품절', '그래서', '심플', '계속', '신으니', '빨라요', '신랑', '물건', '무엇', '사줬는데', '선택', '드립니다', '한번', '곳', '신분', '강추', '작성', '받아', '딸', '여자친구'
            , '지금', '번', '좋음', '확실히', '후회', '봤는데', '전체', '같이', '왜', '있지만', '실제', '굿굿', '나', '굉장히', '기능', '산', '좋아해요', '와서', '예쁘게', '드렸는데', '한데', '같아', '분들', '했지만', '좋아용', '줄', '좋다고'
            , '하시면', '거의', '긴한데', '알았는데', '봄', '온', '백화점', '안드로이드', '내', '신습니다', '이쁨', '아무', '동생', '신을수', '의견', '저', '인해', '도착', '기도', '이유', '짱', '신던', '된', '게시', '만족도', '일반', '주관', '되어'
            , '소지', '무조건', '해도', '미리', '검정색', '사면', '빠르게', '그런데', '제', '무신', '건', '좋구요', '가장', '예쁜', '검은색', '도움', '잘산거', '이렇게', '이쁜데', '산거', '그렇게', '괜찮습니다', '성능', '이쁘', '트', '옥션', '되고'
            , '다닐', '감사', '되서', '오늘', '는', '택배', '시간', '득템', '꿈', '번창', '아니라', '타이', '정품', '딸아이', '없어요', '좋았습니다', '나름', '어느', '캐', '싸', '흰' '모델', '문제', '매일', '임', '남자친구', '시', '살까', '문'
            , '빠르네요', '좋을듯', '했던', '판매', '아니고', '씩', '알', '신을것', '들어하네요', '새', '니', '하지', '될것', '하다', '집', '왔네요', '신고있어요', '두번째', '맨날', '맛', '확인', '재', '가지', '갈', '오프라인', '금방', '다행', '맘에듭니'
            , '어느', '받고', '이쁜거', '특히', '귀엽고', '깔', '있었는데', '걍', '눈', '착한', '인터넷', '이용', '포스', '직접', '이쁘다고', '추합니다', '신경', '싶었는데', '검', '에드', '땐', '달', '신청', '귀찮아서', '상당히', '대박', '사길'
             , '베이지', '어머니', '뭔가', '그거', '리뷰', '적극', '자마자', '아빠', '좋으네요', '만족스러워요', '사실', '첨', '빠릅니다', '찾다가', '하면서', '하시네요', '할게요', '하자', '쫌', '오른쪽', '좋아해서', '좀더', '여러', '작년', '오히려'
            , '재고', '주얼', '착하고', '이런', '편입', '하얀색', '와이프', '좋아하시네요', '일주일', '이틀', '존예', '잘산것', '추강', '감사해요', '조아', '무신사', '예쁘다고', '예상', '짱짱', '빨랐어요', '암튼', '음', '레드', '환불', '해외', '싸게잘'
            , '적립금', '빠름', '좋겠네요', 'G', '예뻐용', '벗', '더욱', '엇', '남친', '지인', '부모님', '여성', '이뿌네요', '취소', '굿굿굿', '노란색', '감사합니당', '행사', '예쁘고요', '무슨', '종일', '해주세요', '괜찮', '바랍니다', '일부러', '딱히'
            , '사랑', '그렇고', '모든', '며칠', '검은', '좋다네요', '새끼', '쇼핑', '굳이', 's', '직원', '정가', '좋아하세요', '언제나', '하긴', '총알', '여러분', '내년', '지급', '언니', '갓', '모르겠어요', '결제', '괜찮음', 'Good', '정확히', '느리지만'
            ,'금액', '빨강', '여러가지', '심하게', '한가지', '당연히', '만원', '예뻐여', 'very', '뭘', '게다가', '갑자기', '아쉬운건', '당장', '부탁드려요', '통해', '고딩', '고등학생', '아영', '일찍', '점점', '완젼', '일요일', '배달', '홈쇼핑', '파시'
            , '출퇴근', '어머님', '스티커', '든다네요', '지연', '좋았구요', '잘쓸게요', '개꿀', '예쁘다', '귀엽습니다', '귀여워서', '물품', '살게요', '왕', '대박나세요', '예쁘다', '말씀', '들어하시네요', '사길잘', '예쁘고요', '오빠', '남동생', '역쉬'
            , '큰일', '화요일', '입학', '노란', '느려서', '기다렸는데', '아닌가', '이상하게', '좋아하시는', '만족하면서', '노란', '사이즈', '이벤트', '딸램', '수업', '블프', '좋아하고', '한국', '근본', '엄청나게', '같네요', '같아용', '가격', '저렴', '싸', '비싸'
            , '싸게', '싸서', '비싸고', '비싸지만', '싸네요', '싸니까', '싸요', '비싸긴', '비싸네요', '싸다', '싸지만', '싸다고', '비싸다고', '비싸지만', '싸구']

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
                     , stopwords=None
                     , delimiter=None
                     , spell_check=True
                     , return_all=False):
        self.k = k
        self.useful_tags=useful_tags
        if stopwords==None:
            self.stopwords=default_stopwords
        else:
            self.stopwords=stopwords
        self.spell_check=spell_check
        self.return_all=return_all
        self.okt = Okt()
        if not isinstance(k, int):
            raise KoreanTextRank4ReviewError('k must be int')
        
        
    def summarize(self, reviews):
        if isinstance(reviews, list):
            reviews = ' '.join(reviews)
        if delimiter == None:
            self.splited_reviews = re.split('\.|\\n|\.\\n|\!', reviews.strip())
        else:
            self.splited_reviews = re.split(delimiter, reviews.strip())
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
                    _checked = _agent.get(base_url, params=payload, headers=headers)
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