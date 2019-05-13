# -*- coding: utf8 -*-
from flask import Flask
from flask import render_template
from flask import request
import json
import numpy as np
import codecs

app = Flask(__name__, template_folder='template/')
from tqdm import tqdm
import random
import sys
reload(sys)
sys.setdefaultencoding('utf8')

random.seed(100)

qa_list = []
with codecs.open("data/raw/dev-v1.1.json", 'r',encoding='utf-8') as f:
    data = json.load(f)
    dev_data = data['data']

with codecs.open("data/prediction.json", 'r',encoding='utf-8') as f:
    predict_dict = json.load(f)

for i in tqdm(range(len(dev_data)), desc="process... dev data"):
    paragraphs = dev_data[i]['paragraphs']
    for paragraph in paragraphs:
        context = paragraph['context']
        context = context.replace("''", '" ').replace("``", '" ').replace('　', ' ')
        qas = paragraph['qas']
        for qa in qas:
            question = qa['question']
            uuid = qa['id']
            answers = []
            for i, ans in enumerate(qa['answers']):
                answers.append(qa['answers'][i]['text'])

            entry = (uuid, context, question, " ✔️ ".join(answers))
            qa_list.append(entry)


def percent(prob_list):
    min = np.min(prob_list)
    probs_ = [prob - min for prob in prob_list]
    sum = np.sum(probs_)
    probs = [round(prob / sum, 4) for prob in probs_]
    print(probs)
    print(max(probs))
    return probs


def render_span(tokens, probs,color="9, 220, 123"):
    # 9, 132, 220 蓝
    # 9, 220, 123 青
    start_spans = ('<span style=" ' \
                   'background-color:rgba({},{:.2f}); ' \
                   'font-size:{}px;' \
                   'border-radius:5px;">{}</span> '.format(color, 0.1 + start_prob, 13 + start_prob * 20, token) for
                   token, start_prob in zip(tokens, probs))
    return ' '.join(start_spans)


@app.route('/')
@app.route('/<int:num>')
def home(num=None):
    # i = 1945 ,2759可以举例

    total_num = len(qa_list)
    i = random.randint(0, total_num - 1) if not num or num >= total_num else num
    entry = qa_list[i]

    return render_template('home.html', index=i, context=entry[1], question=entry[2], answer=entry[3], predict="")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    index = str(request.form['index']).strip()
    context = str(request.form['context']).strip()
    question = str(request.form['question']).strip()
    answer = str(request.form['answer']).strip()
    entry = qa_list[int(index)]

    # predict_entry = predict_dict.get(entry[0] + "\n", None)
    predict_entry = predict_dict.get(entry[0], None)

    # 跳过文章长度过长
    if predict_entry is None:
        predict = "没有预测答案，可能原因：文章长度过长或者有特殊token"
        return render_template('home.html', index=index, context=context, context_prob="", question=question, answer=answer,
                               predict=predict)

    predict_answer = predict_entry['answer']
    context_tokens = predict_entry['tokens']
    context_len = len(context_tokens)
    start_probs = predict_entry['start_probs'][:context_len]
    end_probs = predict_entry['end_probs'][:context_len]

    start_probs = percent(start_probs)
    end_probs = percent(end_probs)

    # 渲染 html
    context_start_prob_html = render_span(context_tokens, start_probs,color="220,9,9")
    context_end_prob_html = render_span(context_tokens, end_probs,color="9,132,220")

    print("id:", index)
    print(entry[0])
    print(repr(predict_answer))

    return render_template('home.html', index=index, context=context,
                           context_start_prob=context_start_prob_html, context_end_prob=context_end_prob_html,
                           question=question, answer=answer, predict=predict_answer)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
