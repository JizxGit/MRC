# _*_coding:utf-8 _*_
import numpy as np
import tensorflow as tf
import nltk

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('ckpt_path', 'model/model.ckpt-100000', '''模型保存路径''')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, '''初始学习率''')
tf.app.flags.DEFINE_integer('train_steps', 50000, '''总的训练轮数''')

import spacy

# 加载英文模型数据，稍许等待
nlp = spacy.load('en_core_web_sm')
#
# def main(args):
#     print(FLAGS.learning_rate, FLAGS.learning_rate)
#
# if __name__ == '__main__':
#     tf.app.run()
#
# def tokenize(sent):
#     tokens = nltk.word_tokenize(sent)
#     return [token.replace("``", '"').replace("''", '"').lower() for token in tokens]
#     # return [token.lower() for token in nltk.word_tokenize(sent)]
#
# def getcharloc2wordloc(context, context_tokens):
#     acc = ''
#     word_index = 0  # word 在文本中 的下标
#     mapping = {}
#     context_token = context_tokens[word_index]
#     # if context.startswith('the code itself was patterned so '):
#     #     pass
#     for i, char in enumerate(context):
#         mapping[i] = (context_token, word_index)  # (word,word下标)
#         char = char.strip() # 去除一些奇怪的不可见字符
#         if len(char) > 0:
#             # if char != u' ' and char != '\n' and char!=u'　' and char!=' ' and char!=' ':
#             acc += char
#             context_token = context_tokens[word_index]
#             if acc == context_token:
#                 word_index += 1
#                 acc = ''
#     if len(context_tokens) != word_index:
#         print(' '.join(context_tokens))
#         print(context)
#         return None
#
#     return mapping

# context="Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50."
# # context="super bowl 50 was an american football game to determine the champion of the national football league ( nfl ) for the 2015 season . the american football conference ( afc ) champion denver broncos defeated the national football conference ( nfc ) champion carolina panthers 24–10 to earn their third super bowl title . the game was played on february 7 , 2016 , at levi 's stadium in the san francisco bay area at santa clara , california . as this was the 50th super bowl , the league emphasized the \" golden anniversary \" with various gold-themed initiatives , as well as temporarily suspending the tradition of naming each super bowl game with roman numerals ( under which the game would have been known as \" super bowl l \" ) , so that the logo could prominently feature the arabic numerals 50 ."
# context="the panthers finished the regular season with a 15–1 record , and quarterback cam newton was named the nfl most valuable player ( mvp ) . they defeated the arizona cardinals 49–15 in the nfc championship game and advanced to their second super bowl appearance since the franchise was founded in 1995 . the broncos finished the regular season with a 12–4 record , and denied the new england patriots a chance to defend their title from super bowl xlix by defeating them 20–18 in the afc championship game . they joined the patriots , dallas cowboys , and pittsburgh steelers as one of four teams that have made eight appearances in the super bowl ."
# context=context.split()
# print(context[29 :30+1])
#
# context = context.replace("''", '" ').replace("``", '" ').replace('　', ' ')
# context_tokens = tokenize(context)  # token是小写的
# context = context.lower()
# charloc2wordloc = getcharloc2wordloc(context, context_tokens)
#
# ans_char_start= 403
# answer= "Santa Clara, California."
# ans_char_end = ans_char_start + len(answer)
# ans_word_start = charloc2wordloc[ans_char_start][1]
# ans_word_end = charloc2wordloc[ans_char_end - 1][1]
# answer_tokens = context_tokens[ans_word_start: ans_word_end + 1]
# answer_span = [str(ans_word_start), str(ans_word_end)]
# print(ans_word_start)
# print(ans_word_end)
# print(charloc2wordloc[ans_char_end])
# print(answer_tokens)

# stack中的axis就是表示对数组 arr 中的第几维的元素进行打包（用中括号[]进行元素打包）
