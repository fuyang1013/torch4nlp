# 2022-11-23
import math
from ..metric import cosine_similarity


class TFIDF:
    """ 实现TFIDF """
    def __init__(self, corpus, stop_words=None):
        """
        corpus(List[List[str]]): 语料库，格式为列表；语料库包含若干个长度不一的文档，每个文档为一个列表，元素为切好的词
        stop_words(iterable): 停用词表，用于过滤不重要的词
        """
        
        self.word2idf = self._get_word2idf(corpus, stop_words)
    
    def _get_word2idf(self, corpus, stop_words):
        """构建词 -> IDF的词典
        """
        
        if stop_words is None:
            stop_words = set()
        else:
            stop_words = set(stop_words)

        num_doc = len(corpus)

        word2num_doc_covered = {}  # 词 -> 包含该词的文档
        for doc in corpus:
            uniq_words_in_cur_doc = set(doc)
            for word in uniq_words_in_cur_doc:
                word2num_doc_covered[word] = word2num_doc_covered.get(word, 0) + 1

        word2idf = {}
        for word, freq in word2num_doc_covered.items():
            assert freq > 0, f'{word} not existed in any doc'
            if word not in stop_words:
                word2idf[word] = math.log(num_doc / freq)
        
        return word2idf
    
    def get_tfidf_vec(self, word_list):
        """返回tfidf向量
        """

        if not word_list:
            return [0]*len(self.word2idf)

        word2tf = {}
        for word in word_list:
            word2tf[word] = word2tf.get(word, 0) + 1
        for k in word2tf:
            word2tf[k] = word2tf[k] / len(word_list)
        
        result_vec = []
        for word, idf in self.word2idf.items():
            if word in word2tf:
                result_vec.append(word2tf[word] * idf)
            else:
                result_vec.append(0)
        
        return result_vec

    def get_top_doc(self, word_list, distance_method='cosine_sim', topk=1):
        """返回语料库中相似度最高的
        """

        


