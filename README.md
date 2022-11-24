# KoreanReviewSummarizer
textrank, pagerank와 bm25를 이용한 한국어 리뷰 요약 python 패키지입니다.
## Run on Virtualenv Environment
```sh
pip install virtualenv

cd [Project_Directory]
virtualenv [Virtualenv_Name]
source [Virtualenv_Name]/bin/activate
# (to exit venv) deactivate
```
## Installation
```sh
pip install ks4r
pip install scipy
```
## Settings
```python
# in __init__.py file

from ks4r import Summarizer
summarizer = Summarizer() #초기화
summary = summarizer.summarize(text) #여러 리뷰들을 하나의 String으로 만들어 넣으시면 됩니다.
```
## Default parameter
```python
summarizer = Summarizer(k=3
                      , useful_tags=['Noun', 'Verb', 'Adjective', 'Determiner', 'Adverb', 'Conjunction', 'Josa', 'PreEomi', 'Eomi', 'Suffix', 'Alpha', 'Number']
                      , stopwords=None
	          , delimiter='\.|\\n|\.\\n|\!'
                      , spell_check=True
                      , return_all=False)
k=3 #반환할 요약 문장의 갯수 입니다
useful_tags=['Noun', 'Verb', 'Adjective', 'Determiner', 'Adverb', 'Conjunction', 'Josa', 'PreEomi', 'Eomi', 'Suffix', 'Alpha', 'Number'] #가중치 계산에 사용할 형태소입니다.
stopwords=stopwords #리뷰 분석에 불필요한 불용어 목록입니다. 리뷰에 맞추어 list형식으로 만들어 적용하시면 됩니다. 기본은 쇼핑몰 리뷰에 맞추어져있습니다.
delimiter='\.|\\n|\.\\n|\!' #문장 구분자를 입력합니다.
spell_check=True #네이버 맞춤법 검사기를 사용합니다. 성능이 좋아지지만 시간이 오래걸립니다.
return_all=False #True인 경우 k에 상관없이 들어간 모든 리뷰의 문장들을 pagerank score가 높은순으로 반환합니다.
```
