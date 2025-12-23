import streamlit as st
import jieba
from collections import Counter
import re
from snownlp import SnowNLP

# 页面配置
st.set_page_config(page_title="文本分析工具", page_icon="📝", layout="wide")

# 标题与说明
st.title("📝 文本分析Web应用")
st.markdown("支持**中文分词、词频统计、情感分析、关键词提取**")

# 文本输入区域
text = st.text_area("请输入待分析的中文文本", height=200, placeholder="例如：今天天气很好，我很开心！")

# 侧边栏功能选择
st.sidebar.title("功能设置")
stop_words = st.sidebar.checkbox("过滤停用词", value=True)
top_n = st.sidebar.slider("显示TOP词频数量", 5, 30, 10)

# 停用词列表（基础版）
STOP_WORDS = set([
    '的', '了', '是', '我', '你', '他', '她', '它', '们', '在', '和', '有', '就', '不', '也', '还', '这', '那',
    '着', '过', '啊', '呀', '哦', '呢', '吧', '吗', '哈', '哎', '嗯', '一个', '一些', '什么', '怎么', '哪里'
])

# 文本预处理函数
def preprocess_text(text):
    # 去除标点、数字和空格
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    # 分词
    words = jieba.lcut(text)
    # 过滤停用词和单字
    if stop_words:
        words = [word for word in words if word not in STOP_WORDS and len(word) > 1]
    return words

# 分析逻辑
if text:
    with st.spinner("正在分析文本..."):
        # 1. 文本基础信息
        st.subheader("1. 文本基础信息")
        col1, col2, col3 = st.columns(3)
        col1.metric("总字符数", len(text))
        words = preprocess_text(text)
        col2.metric("有效分词数", len(words))
        col3.metric("唯一词数", len(set(words)))

        # 2. 词频统计
        st.subheader("2. 词频统计（TOP{}）".format(top_n))
        word_freq = Counter(words).most_common(top_n)
        # 展示词频表格
        st.table(word_freq)
        # 词频柱状图
        st.bar_chart({k: v for k, v in word_freq})

        # 3. 情感分析
        st.subheader("3. 情感分析")
        s = SnowNLP(text)
        sentiment_score = s.sentiments
        sentiment = "积极" if sentiment_score > 0.5 else "消极" if sentiment_score < 0.5 else "中性"
        col1, col2 = st.columns(2)
        col1.metric("情感得分", round(sentiment_score, 3))
        col2.metric("情感倾向", sentiment)

        # 4. 关键词提取（基于TF-IDF简化版）
        st.subheader("4. 关键词提取")
        keywords = [word for word, freq in word_freq[:5]]  # 取词频TOP5作为关键词
        st.write("提取的关键词：", "、".join(keywords))
else:
    st.info("请在上方输入文本后开始分析")
