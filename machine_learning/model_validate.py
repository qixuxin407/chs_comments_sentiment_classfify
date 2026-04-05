# encoding:utf-8
import os
import joblib
from collections import Counter
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc, confusion_matrix
# 可视化相关
import matplotlib.pyplot as plt
import numpy as np
# 导入通用函数（需保证tokenizer.py在同级目录）
from tokenizer import load_stopwords, jieba_tokenizer

# 路径配置（与训练脚本保持一致）
validate_corpus = r"D:\chs_comments_sentiment_classify\data\chs-comments\test\corpus_validation.txt"  # 测试集语料库
validate_result = r"D:\chs_comments_sentiment_classify\machine_learning\model_results\corpus_validate_results.txt" # 分类结果保存位置
label_file = r"D:\chs_comments_sentiment_classify\data\chs-comments\label\test_validation_label.txt"  # 测试集标签文件
prediction_file = r"D:\chs_comments_sentiment_classify\machine_learning\model_results\corpus_validate_results.txt"  # 预测结果文件
stopwords_path = r"D:\chs_comments_sentiment_classify\dict\stopwords.txt"  # 停用词库路径
# 模型加载路径（需与训练脚本的保存路径一致）
MODEL_SAVE_DIR = r"D:\chs_comments_sentiment_classify\saved_model"
CLASSIFIER_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "sentiment_classifier.joblib")
MLB_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "mlb_encoder.joblib")
# 可视化图片保存路径
PLOT_SAVE_DIR = r"D:\chs_comments_sentiment_classify\machine_learning\model_results"
if not os.path.exists(PLOT_SAVE_DIR):
    os.makedirs(PLOT_SAVE_DIR)
# ====================== 加载停用词和分词函数（与训练脚本保持一致） ======================
# 加载停用词（全局变量）
STOPWORDS = load_stopwords(stopwords_path)


# 必须与训练脚本完全一致的分词函数（模型序列化依赖）
def custom_tokenizer(text):
    """自定义分词函数（全局作用域，支持pickle序列化）"""
    return jieba_tokenizer(text, STOPWORDS)

# ====================== 加载模型并执行预测 ======================
def load_trained_model():
    """加载已训练好的分类器和标签编码器"""
    # 检查模型文件是否存在
    if not os.path.exists(CLASSIFIER_SAVE_PATH):
        raise FileNotFoundError(f"分类器模型文件不存在：{CLASSIFIER_SAVE_PATH}")
    if not os.path.exists(MLB_SAVE_PATH):
        raise FileNotFoundError(f"标签编码器文件不存在：{MLB_SAVE_PATH}")

    # 加载模型
    print(f"正在加载模型：{CLASSIFIER_SAVE_PATH}")
    classifier = joblib.load(CLASSIFIER_SAVE_PATH)
    mlb = joblib.load(MLB_SAVE_PATH)
    print("模型加载完成！")
    return classifier, mlb


def predict_validation_set(classifier, mlb):
    """读取验证集并执行预测，保存预测结果"""
    # 读取验证集语料
    corpus_list = []
    with open(validate_corpus, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                corpus_list.append(line)

    # 执行预测
    print(f"验证集有效样本数(条)：{len(corpus_list)}")
    prediction = classifier.predict(corpus_list)
    result = mlb.inverse_transform(prediction)

    # 保存预测结果
    with open(validate_result, 'w', encoding='utf-8') as f:
        for i in range(len(corpus_list)):
            f.write(corpus_list[i])
            f.write(str(result[i][0]) + '\n')

    # 输出预测分布
    num_dict = Counter(result)
    total = len(result)
    accuracy_rough = (num_dict.get(('0',), 0) + num_dict.get(('1',), 0)) / float(total)
    print(f"预测结果分布：负向({('0',)})={num_dict.get(('0',), 0)}，正向({('1',)})={num_dict.get(('1',), 0)}")
    print(f"有效预测占比：{accuracy_rough:.4f}")
    return result


# ====================== 计算验证指标 ======================
def calculate_validation_metrics():
    """计算并输出验证集的准确率、精确率、召回率、F1值"""
    y_true = []
    y_pred = []

    # 读取真实标签
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                value = int(float(line))
                y_true.append(value)

    # 读取预测标签
    with open(prediction_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                # 取行最后一个字符作为预测标签（与训练脚本逻辑一致）
                value = int(line[-1])
                y_pred.append(value)

    # 校验标签长度
    if len(y_true) != len(y_pred):
        print(f"警告：真实标签数({len(y_true)})与预测标签数({len(y_pred)})不一致！")
        # 截断到较短长度，避免计算报错
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

    # 计算指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)

    # 输出结果
    print("\n===== 验证集指标 =====")
    print(f"准确率(Accuracy)：{accuracy}")
    print(f"精确率(Precision)：{precision}")
    print(f"召回率(Recall)：{recall}")
    print(f"F1值(F1-Score)：{f1}")
    print("=======================")

    # ========== 新增：返回真实标签和预测标签，供可视化使用 ==========
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "y_true": y_true,
        "y_pred": y_pred
    }
    # ==============================================================


# ========== 新增：可视化函数 ==========
def plot_model_metrics(metrics_dict):
    """绘制模型效果可视化图形并保存"""
    # 设置中文字体（避免中文乱码）
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    y_true = metrics_dict['y_true']
    y_pred = metrics_dict['y_pred']
    accuracy = metrics_dict['accuracy']
    precision = metrics_dict['precision']
    recall = metrics_dict['recall']
    f1 = metrics_dict['f1']

    # 1. 混淆矩阵热力图
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    # 归一化（可选）
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im = plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im)
    # 设置标签
    classes = ['负向(0)', '正向(1)']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)
    plt.xlabel('预测标签', fontsize=14)
    plt.ylabel('真实标签', fontsize=14)
    plt.title('混淆矩阵（归一化）', fontsize=16)
    # 标注数值
    thresh = cm_normalized.max() / 2.
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            plt.text(j, i, f'{cm_normalized[i, j]:.2f}\n(绝对值:{cm[i, j]})',
                     horizontalalignment="center",
                     color="white" if cm_normalized[i, j] > thresh else "black",
                     fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_SAVE_DIR, 'confusion_matrix.png'), dpi=300)
    plt.close()

    # 2. 分类指标柱状图
    plt.figure(figsize=(10, 6))
    metrics = ['准确率', '精确率', '召回率', 'F1值']
    values = [accuracy, precision, recall, f1]
    bars = plt.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    # 标注数值
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.4f}', ha='center', va='bottom', fontsize=12)
    plt.ylim(0, 1.1)  # y轴范围0-1.1
    plt.xlabel('评估指标', fontsize=14)
    plt.ylabel('指标值', fontsize=14)
    plt.title('模型分类指标', fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_SAVE_DIR, 'classification_metrics.png'), dpi=300)
    plt.close()

    # 3. 预测结果分布饼图
    plt.figure(figsize=(8, 8))
    pred_counts = Counter(y_pred)
    labels = ['负向(0)', '正向(1)']
    sizes = [pred_counts.get(0, 0), pred_counts.get(1, 0)]
    colors = ['#ff9999', '#66b3ff']
    explode = (0.05, 0.05)  # 分离切片
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.2f%%',
            shadow=True, startangle=90, textprops={'fontsize': 12})
    plt.axis('equal')  # 保证饼图为正圆形
    plt.title('预测结果分布', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_SAVE_DIR, 'prediction_distribution.png'), dpi=300)
    plt.close()

    # 4. ROC曲线+AUC值（二分类专用）
    plt.figure(figsize=(8, 6))
    # 若模型支持predict_proba，用概率计算ROC；否则用预测标签（效果稍差）
    try:
        # 重新加载模型获取预测概率（若分类器支持）
        classifier, _ = load_trained_model()
        # 读取验证集语料用于概率预测
        corpus_list = []
        with open(validate_corpus, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    corpus_list.append(line)
        corpus_list = corpus_list[:len(y_true)]  # 对齐长度
        y_score = classifier.predict_proba(corpus_list)[:, 1]  # 取正向类概率
    except:
        # 若模型不支持predict_proba，用预测标签替代
        y_score = y_pred
        print("提示：分类器不支持predict_proba，ROC曲线使用预测标签替代概率值")

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    # 绘制ROC曲线
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # 随机猜测线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率(FPR)', fontsize=14)
    plt.ylabel('真阳性率(TPR)', fontsize=14)
    plt.title('ROC曲线', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_SAVE_DIR, 'roc_curve.png'), dpi=300)
    plt.close()

    print(f"\n可视化图片已保存至：{PLOT_SAVE_DIR}")
    print("生成的图片：1. confusion_matrix.png - 混淆矩阵热力图 2. classification_metrics.png - 分类指标柱状图 3. prediction_distribution.png - 预测结果分布饼图 4. roc_curve.png - ROC曲线+AUC值")

    return roc_auc #返回auc值
# ====================== 新增：保存模型指标到文件的函数 ======================
def save_model_metrics(metrics_dict, roc_auc):
    """
    保存模型核心指标到model_results.txt文件
    :param metrics_dict: 包含准确率、精确率、召回率、F1值的字典
    :param roc_auc: ROC曲线的AUC值
    """
    # 拼接指标保存路径
    metrics_save_path = os.path.join(PLOT_SAVE_DIR, "model_results.txt")

    # 整理要保存的指标内容
    metrics_content = [
        "===== 中文情感分类模型验证指标 =====",
        f"准确率(Accuracy)：{metrics_dict['accuracy']:.6f}",
        f"精确率(Precision)：{metrics_dict['precision']:.6f}",
        f"召回率(Recall)：{metrics_dict['recall']:.6f}",
        f"F1值(F1-Score)：{metrics_dict['f1']:.6f}",
        f"AUC值：{roc_auc:.6f}",
        "====================================",
        # 补充混淆矩阵信息（可选，增强可读性）
        "\n===== 混淆矩阵（真实标签×预测标签） ====="
    ]

    # 计算混淆矩阵并添加到内容中
    cm = confusion_matrix(metrics_dict['y_true'], metrics_dict['y_pred'])
    metrics_content.append(f"负向(0)预测为负向(0)：{cm[0][0]}  |  负向(0)预测为正向(1)：{cm[0][1]}")
    metrics_content.append(f"正向(1)预测为负向(0)：{cm[1][0]}  |  正向(1)预测为正向(1)：{cm[1][1]}")

    # 写入文件
    with open(metrics_save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(metrics_content))

    print(f"\n模型指标已保存至：{metrics_save_path}")

# ====================== 主函数 ======================
if __name__ == '__main__':
    try:
        # 1. 加载训练好的模型
        classifier, mlb = load_trained_model()

        # 2. 对验证集执行预测
        predict_validation_set(classifier, mlb)

        # 3. 计算并输出验证指标
        metrics_result = calculate_validation_metrics()

        # 4.调用可视化函数
        roc_auc = plot_model_metrics(metrics_result)

        # 5.调用保存指标函数
        save_model_metrics(metrics_result, roc_auc)

    except Exception as e:
        print(f"验证过程出错：{str(e)}")