# encoding:utf-8
import os
import sys
import joblib
# 必须导入和训练时一致的工具函数
sys.path.append(r"D:\chs_comments_sentiment_classify")  # 项目根目录
from utils import load_stopwords, jieba_tokenizer

# ===================== 关键：定义和训练时完全一致的分词函数 =====================
# 1. 加载停用词（路径和训练时保持一致）
STOPWORDS_PATH = r"D:\chs_comments_sentiment_classify\dict\stopwords.txt"
STOPWORDS = load_stopwords(STOPWORDS_PATH)


# 2. 定义和训练脚本中完全同名的全局函数（函数名、参数、逻辑必须一致）
def custom_tokenizer(text):
    """自定义分词函数（必须和train_and_validate.py中完全一致）"""
    return jieba_tokenizer(text, STOPWORDS)


# ===================== 模型加载路径配置 =====================
MODEL_SAVE_DIR = r"D:\chs_comments_sentiment_classify\saved_model"
CLASSIFIER_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "sentiment_classifier.joblib")
MLB_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "mlb_encoder.joblib")


# ===================== 模型加载函数 =====================
def load_model():
    """加载训练好的分类器和标签编码器"""
    try:
        # 检查模型文件是否存在
        if not os.path.exists(CLASSIFIER_SAVE_PATH):
            raise FileNotFoundError(f"分类器模型文件不存在：{CLASSIFIER_SAVE_PATH}")
        if not os.path.exists(MLB_SAVE_PATH):
            raise FileNotFoundError(f"标签编码器文件不存在：{MLB_SAVE_PATH}")

        # 加载模型（此时能找到custom_tokenizer函数）
        classifier = joblib.load(CLASSIFIER_SAVE_PATH)
        mlb = joblib.load(MLB_SAVE_PATH)
        print("模型加载成功！")
        return classifier, mlb

    except Exception as e:
        print(f"模型加载失败：{str(e)}")
        raise e  # 抛出异常，让API服务感知加载失败

# 测试函数
if __name__ == '__main__':
    classifier, mlb = load_model()
    test_text = input("请输入测试文本：")
    pred = classifier.predict([test_text])
    label = mlb.inverse_transform(pred)[0][0]
    print(f"文本：{test_text} → 预测标签：{label}")