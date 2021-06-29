#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gensim.models as g
from gensim.corpora import WikiCorpus#导入维基百科
import logging
from GensimWikiWordVector.doc2vec训练与相似度计算.langconv import *
'******************doc2vec模型进行训练*************************'

#enable logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

docvec_size=400
#tags:doc2vec训练时采用tag信息，更好的辅助训练
class TaggedWikiDocument(object):
    def __init__(self, wiki):
        self.wiki = wiki#传入的是维基百科语料库对象
        self.wiki.metadata = True#元数据中需要添加标签信息

    #迭代器__iter__(self)
    def __iter__(self):
        import jieba
        #wiki.get_texts()获取维基百科的文档，content文档的内容，page_id 文档添加一个id, title其实是文档里面的关键字
        for content, (page_id, title) in self.wiki.get_texts():
            yield g.doc2vec.TaggedDocument(words=[w for c in content for w in jieba.cut(Converter('zh-hans').convert(c))], tags=[title])
        #yield生成器，避免内存不够
def my_function():

    zhwiki_name = 'G:/alldatas/维基百科2019语料库/zhwiki-latest-pages-articles.xml.bz2'
    # zhwiki_name = 'G:/alldatas/维基百科数据的处理/第一步数据下载维基百科 和文本抽取/zhwiki-20201120-pages-articles1.xml-p1p187712.bz2'
    wiki = WikiCorpus(zhwiki_name, lemmatize=False, dictionary={})
    documents = TaggedWikiDocument(wiki)

    #window:其实就是当前词和预测值，可能得到的最大值，
    #dm=0表示不使用DM（分布式记忆模型），而是使用DBow（分布式词袋模型）训练词向量
    model = g.Doc2Vec(documents, dm=0, dbow_words=1, vector_size=docvec_size, window=5, min_count=5, epochs=5, workers=8)
    model.save('F:/python数据模型/doc2vecbuildVector/zhiwiki_news.doc2vec')

if __name__ == '__main__':
    my_function()

