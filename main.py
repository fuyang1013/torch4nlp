import jieba


for line in open('corpus.data'):
    print(jieba.lcut(line.rstrip()))
