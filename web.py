# -*- coding: utf8 -*-
from flask import Flask
from flask import render_template
from flask import request, jsonify
import json
import numpy as np
import codecs

app = Flask(__name__, template_folder='template/')
from tqdm import tqdm
import random
import sys

reload(sys)
sys.setdefaultencoding('utf8')

random.seed(99)

############################### 加载预测结果、验证集文件 ###############################
passage_list = []
with codecs.open("data/raw/dev-v1.1.json", 'r', encoding='utf-8') as f:
    data = json.load(f)
    dev_data = data['data']

with codecs.open("data/prediction.json", 'r', encoding='utf-8') as f:
    predict_dict = json.load(f)

for i in tqdm(range(len(dev_data)), desc="process... dev data"):
    paragraphs = dev_data[i]['paragraphs']
    for paragraph in paragraphs:
        context = paragraph['context'].replace("''", '" ').replace("``", '" ').replace('　', ' ')

        qas = paragraph['qas']
        question_answers = []
        for qa in qas:
            uuid = qa['id']
            question = qa['question']
            answers = []
            for i, ans in enumerate(qa['answers']):
                answers.append(qa['answers'][i]['text'])

            question_answers.append((uuid, question, " ✔️ ".join(answers)))

        passage_list.append({"context": context, "qas": question_answers})


############################### 辅助函数 ###############################

def percent(prob_list):
    ''' 将 softmax 得到的结果 在进行计算百分比，用于显示字体大小 '''
    min = np.min(prob_list)
    sum = np.sum(prob_list) - len(prob_list)*min
    probs = [round(prob - min / sum, 4) for prob in prob_list]
    print(probs)
    print(max(probs))
    return probs


def render_span(tokens, probs, color="9, 220, 123"):
    ''' 渲染显示概率的 html'''
    # 9, 132, 220 蓝
    # 9, 220, 123 青
    start_spans = ('<span style=" ' \
                   'background-color:rgba({},{:.2f}); ' \
                   'font-size:{}px;' \
                   'border-radius:5px;">{}</span> '.format(color, 0.1 + prob, 13 + prob * 20, token) for
                   token, prob in zip(tokens, probs))
    return ' '.join(start_spans)


############################### API ###############################

@app.route('/')
@app.route('/<int:num>')
def home(num=None):
    # 1458 文章可以举例
    # 1570

    total_num = len(passage_list)  # 总数
    context_index = random.randint(0, total_num - 1) if not num or num >= total_num else num  # 随机生成或者指定文章编号
    passage = passage_list[context_index]

    context = passage['context']
    qas = passage['qas']
    question_index = random.randint(0, len(qas) - 1)
    uqa = qas[question_index]

    data = {
        'context_index': context_index,
        'context': context,
        'question_index': question_index,
        'question': uqa[1],
        'qas': qas,
        'true_answer': uqa[2],
        'predict_answer': ""
    }
    return render_template('home.html', **data)


@app.route('/predict', methods=['POST'])
def predict():
    predict_entry = predict_dict.get(request.form['uuid'], None)
    if predict_entry is None:
        result = {
            'predict': "没有预测答案，可能原因：文章长度过长或者有特殊token"
        }
    else:
        predict_answer = predict_entry['answer']
        context_tokens = predict_entry['tokens']
        context_len = len(context_tokens)
        start_probs = predict_entry['start_probs'][:context_len]
        end_probs = predict_entry['end_probs'][:context_len]

        start_probs = percent(start_probs)
        end_probs = percent(end_probs)
        # 渲染 html
        context_start_prob_html = render_span(context_tokens, start_probs)
        context_end_prob_html = render_span(context_tokens, end_probs, color="9,132,220")

        result = {
            'success': "true",
            'context_start_prob': context_start_prob_html,
            'context_end_prob': context_end_prob_html,
            'predict_answer': predict_answer
        }
    return jsonify(result)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
