# encoding:utf-8
#通用函数：加载停用词、加载分词函数
import jieba

# 加载停用词库
def load_stopwords(stopwords_path):
    """加载停用词库，返回停用词集合"""
    stopwords = set()
    try:
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # 跳过空行
                    stopwords.add(line)
        print(f"成功加载停用词库，共{len(stopwords)}个停用词")
    except FileNotFoundError:
        print(f"警告：未找到停用词文件 {stopwords_path}，将跳过停用词过滤")
    return stopwords

# 分词函数：添加停用词过滤
def jieba_tokenizer(x, stopwords):
    """结巴分词 + 停用词过滤"""
    # 分词
    words = jieba.cut(x, cut_all=True)
    # 过滤停用词 + 过滤空字符串
    filtered_words = [word for word in words if word.strip() and word not in stopwords]
    return filtered_words