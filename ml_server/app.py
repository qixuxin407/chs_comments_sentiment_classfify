# encoding:utf-8
import os
import sys
from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS  # 新增导入cors跨域传递

sys.path.append(r"D:\chs_comments_sentiment_classify")  # 项目根目录
from utils import load_stopwords, jieba_tokenizer

# ===================== 全局配置 =====================
app = Flask(__name__)
CORS(app)

# 1. 路径配置（和训练/加载脚本保持一致）
STOPWORDS_PATH = r"D:\chs_comments_sentiment_classify\dict\stopwords.txt"
MODEL_SAVE_DIR = r"D:\chs_comments_sentiment_classify\saved_model"
CLASSIFIER_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "sentiment_classifier.joblib")
MLB_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "mlb_encoder.joblib")

# 2. 加载停用词（全局仅加载一次）
STOPWORDS = load_stopwords(STOPWORDS_PATH)


# 3. 定义和训练/加载脚本完全一致的分词函数（必须全局定义）
def custom_tokenizer(text):
    """自定义分词函数：和train_and_validate.py、model_loader.py完全一致"""
    return jieba_tokenizer(text, STOPWORDS)


# ===================== 全局加载模型（启动时加载一次，避免每次请求重新加载） =====================
try:
    # 加载分类器和标签编码器
    classifier = joblib.load(CLASSIFIER_SAVE_PATH)
    mlb = joblib.load(MLB_SAVE_PATH)
    print("模型加载成功，API服务可正常使用")
except Exception as e:
    print(f"模型加载失败：{str(e)}")
    raise e  # 启动失败，避免服务空跑


# ===================== API接口 =====================
@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    """
    情感分类预测接口
    请求格式：{"text": "需要预测的文本内容"}
    响应格式：{"code": 200, "msg": "success", "data": {"sentiment": "0/1", "text": "输入文本"}}
    """
    try:
        # 1. 获取请求参数
        req_data = request.get_json()
        if not req_data or 'text' not in req_data:
            return jsonify({
                "code": 400,
                "msg": "参数错误：请传入{'text': '待预测文本'}格式的JSON数据",
                "data": None
            })

        text = req_data['text'].strip()
        if not text:
            return jsonify({
                "code": 400,
                "msg": "参数错误：文本内容不能为空",
                "data": None
            })

        # 2. 预测（注意：classifier.predict接收列表格式）
        prediction = classifier.predict([text])  # 传入列表，避免单文本维度错误
        sentiment_label = mlb.inverse_transform(prediction)[0][0]  # 解析标签

        # 3. 返回结果
        return jsonify({
            "code": 200,
            "msg": "success",
            "data": {
                "text": text,
                "sentiment": sentiment_label,  # 0=负向，1=正向
                "sentiment_desc": "正向" if sentiment_label == '1' else "负向"
            }
        })

    except Exception as e:
        # 异常捕获，避免服务崩溃
        return jsonify({
            "code": 500,
            "msg": f"预测失败：{str(e)}",
            "data": None
        })


# ===================== 启动服务 =====================
if __name__ == '__main__':
    # 配置调试模式/端口（根据需要调整）
    app.run(
        host='0.0.0.0',  # 允许外部访问
        port=5000,  # 端口号
        debug=False  # 生产环境关闭debug
    )