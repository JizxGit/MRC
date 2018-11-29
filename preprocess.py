# _*_ coding:utf8 _*_
import json
from tqdm import tqdm
import nltk
import os
import codecs
import sys

reload(sys)
sys.setdefaultencoding('utf8')


def get_data_from_json(file_name, data_type):
    with open(file_name, 'r') as f:
        data = json.load(f)
    return data


def tokenize(sent):
    tokens = nltk.word_tokenize(sent)
    return [token.replace("``", '"').replace("''", '"').lower() for token in tokens]
    # return [token.lower() for token in nltk.word_tokenize(sent)]


def getcharloc2wordloc(context, context_tokens):
    acc = ''
    word_index = 0  # word 在文本中 的下标
    mapping = {}
    context_token = context_tokens[word_index]
    # if context.startswith('the code itself was patterned so '):
    #     pass
    for i, char in enumerate(context):
        mapping[i] = (context_token, word_index)  # (word,word下标)
        char = char.strip()
        if len(char) > 0:
            # if char != u' ' and char != '\n' and char!=u'　' and char!=' ' and char!=' ':
            acc += char
            context_token = context_tokens[word_index]
            if acc == context_token:
                word_index += 1
                acc = ''
    if len(context_tokens) != word_index:
        print(' '.join(context_tokens))
        print(context)
        return None
    return mapping


def preprocess_and_save_data(data_type, file_name, out_dir):
    '''
    将json数据处理为训练数据，并保存
    :param data_type: train\dev
    :param file_name: 文件名
    :param out_dir: 保存位置
    :return:
    '''
    with open(file_name, 'r') as f:
        data = json.load(f)
        dataset = data['data']

    examples = []
    num_mappingprob, num_tokenprob = 0, 0

    for i in tqdm(range(len(dataset)), desc="process... {} data".format(data_type)):
        paragraphs = dataset[i]['paragraphs']
        for paragraph in paragraphs:
            context = paragraph['context']
            context = context.replace("''", '" ').replace("``", '" ').replace('　', ' ')
            context_tokens = tokenize(context)  # token是小写的
            context = context.lower()  # 在tokenize后再小写，因为nltk对大小写敏感

            qas = paragraph['qas']
            charloc2wordloc = getcharloc2wordloc(context, context_tokens)  # char id 映射到 word id
            if charloc2wordloc is None:
                num_mappingprob += len(qas)
                continue

            for qa in qas:
                question = qa['question']
                question_tokens = tokenize(question)

                # 只选择第一个答案
                answer = qa['answers'][0]['text'].lower()
                ans_char_start = qa['answers'][0]['answer_start']
                ans_char_end = ans_char_start + len(answer)
                ans_word_start = charloc2wordloc[ans_char_start][1]
                ans_word_end = charloc2wordloc[ans_char_end - 1][1]
                answer_tokens = context_tokens[ans_word_start: ans_word_end + 1]
                answer_span = [str(ans_word_start), str(ans_word_end)]

                # ans_tokens has "fifth-generation" but the ans_text is "generation", which doesn't match.
                if ''.join(answer_tokens) != ''.join(answer.split()):
                    num_tokenprob += 1
                    continue

                examples.append((' '.join(context_tokens), ' '.join(question_tokens), ' '.join(answer_tokens), ' '.join(answer_span)))
    # TODO 随机打乱
    print(out_dir + data_type)
    with codecs.open(os.path.join(out_dir, data_type) + '.context', 'w', encoding='utf-8') as context_writer, \
            codecs.open(os.path.join(out_dir, data_type) + '.question', 'w', encoding='utf-8') as question_writer, \
            codecs.open(os.path.join(out_dir, data_type) + '.answer', 'w', encoding='utf-8') as answer_writer, \
            codecs.open(os.path.join(out_dir, data_type) + '.span', 'w', encoding='utf-8') as answer_span_writer:

        for i in range(len(examples)):
            context, question, answer, answer_span = examples[i]

            context_writer.write(context.strip() + '\n')
            question_writer.write(question + '\n')
            answer_writer.write(answer + '\n')
            answer_span_writer.write(str(answer_span) + '\n')
    print "Number of (context, question, answer) triples discarded due to char -> token mapping problems: ", num_mappingprob
    print "Number of (context, question, answer) triples discarded because character-based answer span is unaligned with tokenization: ", num_tokenprob
    print("{}/{} examples saved".format(len(examples), len(examples) + num_tokenprob + num_mappingprob))


def main(FLAGS):
    raw_data_dir = FLAGS.raw_data_dir  # 原始json文件目录
    prepro_data_dir = FLAGS.prepro_data_dir  # 预处理后的数据存放目录
    if not os.path.exists(prepro_data_dir):
        os.makedirs(prepro_data_dir)

    train_data = 'train-v1.1.json'
    dev_data = 'dev-v1.1.json'

    train_context_path = os.path.join(FLAGS.prepro_data_dir, "train.context")
    train_ques_path = os.path.join(FLAGS.prepro_data_dir, "train.question")
    train_ans_span_path = os.path.join(FLAGS.prepro_data_dir, "train.span")
    dev_context_path = os.path.join(FLAGS.prepro_data_dir, "dev.context")
    dev_ques_path = os.path.join(FLAGS.prepro_data_dir, "dev.question")
    dev_ans_span_path = os.path.join(FLAGS.prepro_data_dir, "dev.span")

    if not os.path.exists(train_context_path) or not os.path.exists(train_ques_path) or not os.path.exists(train_ans_span_path):
        preprocess_and_save_data('train', os.path.join(raw_data_dir, train_data), prepro_data_dir)

    if not os.path.exists(dev_context_path) or not os.path.exists(dev_ques_path) or not os.path.exists(dev_ans_span_path):
        preprocess_and_save_data('dev', os.path.join(raw_data_dir, dev_data), prepro_data_dir)


if __name__ == '__main__':
    main()
