# _*_ coding:utf8 _*_
import re
import string
from collections import Counter


def normalize_answer(s):
    # 删除冠词
    def remove_articles(text):
        return re.sub(r'(a|an|the)', ' ', text)

    # 删除多余空格
    def remove_redundant_white_space(text):
        return ' '.join(text.split())

    # 删除标点
    def remove_punc(text):
        punc = set(string.punctuation)
        return ''.join([c for c in text if c not in punc])

    def lower(text):
        return text.lower()

    return remove_redundant_white_space(remove_redundant_white_space(remove_punc(lower(s))))


def f1_score(pred_ans, true_ans):
    pred_ans_tokens = normalize_answer(pred_ans).split()
    true_ans_tokens = normalize_answer(true_ans).split()
    common = Counter(pred_ans_tokens) & Counter(true_ans_tokens)
    same_token_num = sum(common.values())
    if same_token_num == 0:
        return 0
    else:
        precision = 1.0 * same_token_num / len(pred_ans_tokens)
        recall = 1.0 * same_token_num / len(true_ans_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

def em_score(pred_ans, true_ans):
    return (normalize_answer(pred_ans) == normalize_answer(true_ans))

def metric_max_over_ground_truths(metric_fn,pred_ans,ground_truths):
    '''
    计算预测的答案与多个真实答案中 评价指标的最高得分
    :param metric_fn:
    :param pred_ans:
    :param ground_truths:
    :return:
    '''
    scores=[]
    for true_ans in ground_truths:
        score=metric_fn(pred_ans,true_ans)
        scores.append(score)
    return max(scores)