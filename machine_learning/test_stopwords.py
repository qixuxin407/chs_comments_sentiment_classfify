# 临时测试代码：对比过滤前后的分词结果
import jieba

# 加载停用词（复用之前的函数）
def load_stopwords():
    stopwords_path = r"D:\chs_comments_sentiment_classify\dict\stopwords.txt"
    stopwords = set()
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                stopwords.add(line)
    return stopwords

STOPWORDS = load_stopwords()

# 取一条真实的微博评论测试
test_text = "这个产品真的太坑了，完全不值这个价！我觉得需要改进，差评！"  # 替换成你的语料中的真实评论
# 原分词（无停用词过滤）
words_original = [w for w in jieba.cut(test_text, cut_all=False) if w.strip()]
# 过滤后分词
words_filtered = [w for w in jieba.cut(test_text, cut_all=False) if w.strip() and w not in STOPWORDS]

print("原分词结果：", words_original)
print("过滤停用词后：", words_filtered)