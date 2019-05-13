# _*_ coding:utf8 _*_
import re
import string
from collections import Counter
import json


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


def metric_max_over_ground_truths(metric_fn, pred_ans, ground_truths):
    '''
    计算预测的答案与多个真实答案中 评价指标的最高得分
    :param metric_fn:
    :param pred_ans:
    :param ground_truths:
    :return:
    '''
    scores = []
    for true_ans in ground_truths:
        score = metric_fn(pred_ans, true_ans)
        scores.append(score)
    return max(scores)
    # return scores[0]


def official_evaluate(dataset, predictions):
    '''
     计算测试数据集的官方评估成绩
    :param dataset: 是原始的 json 数据
    :param predictions: 是模型预测的答案字符串
    :return:
    '''
    f1 = exact_match = total = 0

    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                uuid = qa['id']
                if uuid not in predictions:
                    message = 'Unanswered question ' + uuid + ' will receive score 0.'
                    continue

                total += 1

                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[uuid]
                prediction_text = prediction['answer'] # 获取预测的答案文本
                exact_match += metric_max_over_ground_truths(
                    em_score, prediction_text, ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction_text, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'em': exact_match, 'f1': f1}


def print_test_score():
    # 获取测试的 json 数据，包含有多个正确答案的原始数据
    with open("./data/raw/dev-v1.1.json") as dataset_file:
        dataset_json = json.load(dataset_file)
        dataset = dataset_json['data']

    # 获取模型预测保存的 uuid2ans 的 json 数据
    # with open("./data/prediction.json") as prediction_file:
    with open("./data/prediction.json") as prediction_file:
        predictions = json.load(prediction_file)

    # 使用官方的评价方式进行评价
    result = official_evaluate(dataset, predictions)
    print(result)
    return result


if __name__ == '__main__':
    print_test_score()
