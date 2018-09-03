# coding = utf-8
import re

re_pattern = '(.+)'

## 向左裁剪要求比较多
left_separator_list = ['！', '：', '，', '。',
                       '（', '）', '【', '】',
                       '——', '='
                             '!', ':', ',',
                       '(', ')', '[', ']'
                                      ' ', ]
## 向右裁剪比较宽松，针对逗号，句号等即可
right_separator_list = ['！', '，', '。',
                        '!', ',',
                        ' ', ]


def anti_escape_char(str):
  """
  处理转义字符, 包括$  (  )   *  +   .    [   ]   ?   \   ^   {    }    |
  :return:
  """
  str = str.replace('\\', '\\\\')
  str = str.replace('$', '\$')
  str = str.replace('(', '\(')
  str = str.replace(')', '\)')
  str = str.replace('*', '\*')
  str = str.replace('+', '\+')
  str = str.replace('.', '\.')
  str = str.replace('[', '\[')
  str = str.replace(']', '\]')
  str = str.replace('?', '\?')
  str = str.replace('^', '\^')
  str = str.replace('{', '\{')
  str = str.replace('$', '\$')
  str = str.replace('|', '\|')

  return str


def identify_templete(question,
                      feature_words, stop_words,
                      use_templete_stop_words=False):
  templetes = []
  question = anti_escape_char(question)
  tmp_list = ['？', '。', ' ']
  while question[-1] in tmp_list:
    question = question[:-1]
  # question = question.replace('？', '')  # 去除问号
  # question = question.replace('。', '')  # 去除句号
  question = question.strip()
  for word in feature_words:
    if question.count(word) > 0:
      q_templete = question.replace(word, re_pattern)
      if use_templete_stop_words:
        for word in stop_words:
          q_templete = q_templete.replace(word, '.?' * len(word))
      templetes.append(q_templete)
      # print('match:',word)
      # print(q_templete)

  return templetes


def do_left_shrink(templete):
  left_re_pattern_index = templete.index(re_pattern)
  left_shrink_templete = templete[:left_re_pattern_index]
  for separator in left_separator_list:
    if left_shrink_templete.find(separator) > 0:
      left_shrink_templete = left_shrink_templete[left_shrink_templete.rindex(separator) + 1:]
  left_shrink_templete += templete[left_re_pattern_index:]
  return left_shrink_templete


def do_right_shrink(templete):
  right_re_pattern_index = templete.index(re_pattern) + len(re_pattern)
  right_shrink_templete = templete[right_re_pattern_index:]
  for separator in right_separator_list:
    if right_shrink_templete.find(separator) > 0:
      right_shrink_templete = right_shrink_templete[:right_shrink_templete.index(separator)]
  right_shrink_templete = templete[:right_re_pattern_index] + right_shrink_templete
  return right_shrink_templete


def templete_shrink(templetes):
  """
  以匹配符为中心，以不同的程度收缩匹配模板
  比如：
  '据总统代表哈利罗克称，杜特尔特在当地时间(.+)做出了该停火决定' 如果考虑收缩至第一个‘,’的话可以变成:
  '杜特尔特在当地时间(.+)做出了该停火决定' 或者'.*杜特尔特在当地时间(.+)做出了该停火决定'

  收缩mode:
  1) 向左遇到第一个left_separator
  2) 向右遇到第一个right_separator
  3) 左右大小为n的sliding window
  :param templetes:
  :return:
  """
  # fixme: 需要注意有多个匹配符的情况, 所以向左要第一个匹配符，向右要最后一个匹配符
  # TODO:
  new_templetes = []
  for templete in templetes:
    new_templetes.append(templete)

    ## 每个模板执行左,右收缩:

    left_shrink_templete = do_left_shrink(templete)
    new_templetes.append(left_shrink_templete)

    right_shrink_templete = do_right_shrink(templete)
    new_templetes.append(right_shrink_templete)

  return new_templetes


def match_content(templete, content, feature_words):
  found_ans = re.findall(templete, content)
  # if len(found_ans) > 0:
  #   print('ans,\t',found_ans,'\ttemplete:',templete)
  # else:
  #
  if len(found_ans) > 0:
    found_ans = found_ans[0]
    # 有些时候问题中会出现两个“什么”之类的词，导致find出来的东西是tuple的list
    # 需要手工处理掉其中的一个“什么”； 如果两个词全都是“什么”， 则直接返回None
    if isinstance(found_ans, tuple):
      for ans in found_ans:
        if ans not in feature_words:
          found_ans = ans
          break
    if isinstance(found_ans, tuple):
      return None
    if templete.endswith(re_pattern):
      # TODO: 后面不是连词的前提下才截断
      for separator in right_separator_list:
        if found_ans.find(separator) > 0:
          found_ans = found_ans[:found_ans.index(separator)]
    elif templete.startswith(re_pattern):
      # TODO: 前面不是连词的前提下才截断
      for separator in left_separator_list:
        if found_ans.find(separator) > 0:
          found_ans = found_ans[found_ans.rindex(separator) + 1:]
    else:
      # 在中间的话存在匹配过长的问题:
      # 比如'A抓捕了(.+)名武装分子'可能会匹配到'A抓捕了3名武装分子。XXXXXXXX这些武装分子'
      postfix = templete[templete.index(re_pattern) + len(re_pattern):]
      if found_ans.find(postfix) >= 0:
        post_index = found_ans.index(postfix)
        found_ans = found_ans[:post_index]
    found_ans = found_ans.strip()
    return found_ans
  else:
    return None
