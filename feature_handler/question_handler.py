import re
import numpy as np

TYPE_LIST = [
  'who', 'what', 'where', 'when', 'why', 'how',
  'count', 'topic', 'source',
  'others'
]
TYPE_DICT = {}
for type in TYPE_LIST:
  TYPE_DICT[type] = len(TYPE_DICT)

TYPE_KEYWORDS = {
  'who': ['谁', '什么人', '哪个发言人'],
  'what': ['哪', '什么'],
  'where': ['地点', '何地', '哪里', '何处', '哪片'],
  'when': ['时间', '何时', '何日', '时候'],
  'why': ['缘何', '为何'],
  'how': ['如何', '怎么', '怎样'],

  'count': ['多少', '几', '多久'],
  'topic': ['主旨', '大意', '内容', '态度', '目的', '文章说了什么', '介绍了什么'],
  'source': ['源自', '来源', '作者是'],

  # hidden -- 'others':[]
}


class QuestionTypeHandler(object):
  def __init__(self):
    self.type_count = len(TYPE_LIST)
    pass

  def ana_type(self, question):
    matched_types = []
    for type in TYPE_KEYWORDS:
      for keyword in TYPE_KEYWORDS[type]:
        if question.find(keyword) >= 0:
          matched_types.append(type)
          break
    if len(matched_types) == 0:
      matched_types.append('others')

    if len(matched_types) > 1:
      matched_types = [type for type in matched_types if type != 'what']

    type_vec = np.zeros([self.type_count], dtype=np.float32)
    for type in matched_types:
      type_vec[self.get_type_id(type)] = 1.0

    return matched_types,type_vec

  def get_type_id(self, type):
    # assert type in TYPE_DICT  # 默认一定在TypeDict里面
    return TYPE_DICT[type]


def demo():
  handler = QuestionTypeHandler()
  question = '这篇文章的来源是?'

  print(handler.ana_type(question))


if __name__ == '__main__':
  demo()
