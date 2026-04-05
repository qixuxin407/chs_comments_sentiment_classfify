# encoding:utf-8
import pandas as pd
import jieba
import os  # 导入os模块（创建模型目录用）
import joblib  # 导入模型保存工具
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression
# 导入通用函数
from tokenizer import load_stopwords, jieba_tokenizer

# ====================== 新增：导入GridSearchCV所需模块 ======================
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
# ==========================================================================

# 路径添加r前缀避免转义（\t会被转义为制表符）
pos_corpus = r"D:\chs_comments_sentiment_classify\data\chs-comments\train\pos.txt"  # 正向情感训练语料库
neg_corpus = r"D:\chs_comments_sentiment_classify\data\chs-comments\train\neg.txt"  # 负向情感训练语料库
stopwords_path = r"D:\chs_comments_sentiment_classify\dict\stopwords.txt"  # 停用词库路径
# 模型保存路径配置
MODEL_SAVE_DIR = r"D:\chs_comments_sentiment_classify\saved_model"  # 模型保存根目录
CLASSIFIER_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "sentiment_classifier.joblib")  # 分类器保存路径
MLB_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "mlb_encoder.joblib")  # 标签编码器保存路径
GRID_SEARCH_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "grid_search_result.joblib")  # 新增：保存网格搜索结果

# 加载停用词（全局变量，仅加载一次）
STOPWORDS = load_stopwords(stopwords_path)


# 修复：将lambda改为全局具名函数，支持序列化
def custom_tokenizer(text):
    """自定义分词函数（全局作用域，支持pickle序列化）"""
    return jieba_tokenizer(text, STOPWORDS)


# 获取数据和标记
def load_data():
    # 修复：替换pd.read_table(sep='\n')的错误用法，改用直接读取文件行（更高效且避免报错）
    posting_list = []
    class_list = []  # 方便计算转换为1,2,3

    # 读取负向语料
    with open(neg_corpus, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                posting_list.append(line)
                class_list.append('0')

    # 读取正向语料
    with open(pos_corpus, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                posting_list.append(line)
                class_list.append('1')

    print(class_list)
    return posting_list, class_list


def get_classify():
    X_train, Y_train = load_data()

    # 定义分类器
    classifier = Pipeline([
        ('counter', CountVectorizer(
            tokenizer=custom_tokenizer,  # 替换lambda为全局具名函数
            token_pattern=None)),  # 标记和计数，提取特征用 向量化
        ('tfidf', TfidfTransformer()),  # TF-IDF 权重
        ('clf', OneVsRestClassifier(LinearSVC())),  # 1-rest 多分类(多标签)
    ])
    mlb = MultiLabelBinarizer()
    Y_train = mlb.fit_transform(Y_train)  # 分类号数值化

    # ====================== GridSearchCV超参数调优 ======================
    # 1. 定义超参数网格（可根据需求调整）
    param_grid = {
        # CountVectorizer参数
        'counter__ngram_range': [(1, 1), (1, 2)], # 单字/字+二元词组
        'counter__min_df':[1,2], #去掉低频噪声词
        'counter__max_df':[0.4, 0.5], #去掉太常见的词
        #此处参数调优时，发现刚开始给的几个参数（mindf，maxdf，C值）在调优后贴边了，即在搜索范围内的极值，怀疑搜索空间不够精准不够大或不合理。调整范围后精准度提高到0.81。nice！很高兴！
        # TfidfTransformer参数
        'tfidf__use_idf': [True, False],
        'tfidf__sublinear_tf': [True, False],
        # LinearSVC参数
        'clf__estimator__C': [5.0, 8.0],  # 惩罚系数
        'clf__estimator__max_iter': [1000, 2000]  # 最大迭代次数
    }

    # 2. 定义评分指标（兼顾准确率和召回率）
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'f1': make_scorer(f1_score, average='macro'),
        'recall': make_scorer(recall_score, average='macro',zero_division=1)
    }

    # 3. 初始化GridSearchCV（5折交叉验证，并行计算）
    print("开始超参数网格搜索...")
    grid_search = GridSearchCV(
        estimator=classifier,
        param_grid=param_grid,
        cv=5,  # 5折交叉验证
        scoring=scoring,
        refit='f1',  # 以F1值为最优指标选择模型
        n_jobs=-1,  # 使用所有CPU核心加速
        verbose=1  # 输出调优过程日志
    )

    # 4. 执行网格搜索（替换原classifier.fit）
    grid_search.fit(X_train, Y_train)

    # 5. 输出调优结果
    print("\n===== 超参数调优结果 =====")
    print(f"最佳F1分数: {grid_search.best_score_:.4f}")
    print(f"最佳参数组合: {grid_search.best_params_}")
    print("==========================\n")

    # 6. 替换为最优模型
    classifier = grid_search.best_estimator_
    # ==========================================================================

    # 保存模型核心逻辑（优化：增加覆盖提示）
    # 创建模型保存目录（如果不存在）
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        print(f"创建模型保存目录：{MODEL_SAVE_DIR}")

    # 检查文件是否存在，存在则提示
    if os.path.exists(CLASSIFIER_SAVE_PATH):
        print(f"分类器文件 {CLASSIFIER_SAVE_PATH} 已存在，将覆盖原有文件")
    if os.path.exists(MLB_SAVE_PATH):
        print(f"标签编码器文件 {MLB_SAVE_PATH} 已存在，将覆盖原有文件")
    if os.path.exists(GRID_SEARCH_SAVE_PATH):
        print(f"网格搜索结果文件 {GRID_SEARCH_SAVE_PATH} 已存在，将覆盖原有文件")

    # 保存训练好的分类器和标签编码器
    joblib.dump(classifier, CLASSIFIER_SAVE_PATH)
    joblib.dump(mlb, MLB_SAVE_PATH)
    joblib.dump(grid_search, GRID_SEARCH_SAVE_PATH)  # 保存网格搜索结果，便于后续分析
    print(f"分类器已保存至：{CLASSIFIER_SAVE_PATH}")
    print(f"标签编码器已保存至：{MLB_SAVE_PATH}")
    print(f"网格搜索结果已保存至：{GRID_SEARCH_SAVE_PATH}")

# 主函数
if __name__ == '__main__':
    get_classify()