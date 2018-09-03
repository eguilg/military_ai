# coding = utf-8
"""
回答比如主旨, 文章来源, 谁称XX等问题
"""
from feature_handler.handler import Handler
from utils.rouge import RougeL



class TopicHandler():
  def __init__(self, is_train):
    self.details = {}
    self.details[Handler.CONTENT_TYPE_TITLE] = {
      'times': 0,
      'found_times': 0,
    }
    if is_train:
      self.details[Handler.CONTENT_TYPE_TITLE]['rouge_scores'] = []
      self.rogue_eval = RougeL()

  def ans_question(self, content, question, question_ans=None):
    """
      定向回答主旨类的问题

    :param content:
    :param question:
    :param content_type: 区分article和title (注意：回答主旨时目前默认从title寻找答案)
    :return:
    """
    topic_key_words = ['主旨', '大意', '内容', '文章说了什么', '介绍了什么']
    for key in topic_key_words:
      if question.find(key) >= 0:
        if (question.find('文') >= 0 or question.find('报道') > 0) and len(question) <= 12:
          pred_ans = content
          if pred_ans.strip().endswith(')'):
            try:
              pred_ans = pred_ans[:pred_ans.rindex('(')] + pred_ans[pred_ans.rindex(')') + 1:]
            except:
              pass
          elif pred_ans.strip().endswith('）'):
            try:
              pred_ans = pred_ans[:pred_ans.rindex('（')] + pred_ans[pred_ans.rindex('）') + 1:]
            except:
              pass

          if question_ans is not None:
            self.record_found(pred_ans, question_ans)
          return pred_ans
    self.record_miss()
    return None

  def record_found(self, pred_ans, gt_ans):
    self.details[Handler.CONTENT_TYPE_TITLE]['times'] += 1
    self.details[Handler.CONTENT_TYPE_TITLE]['found_times'] += 1
    score = self.rogue_eval.calc_score(pred_ans, gt_ans)
    self.details[Handler.CONTENT_TYPE_TITLE]['rouge_scores'].append(score)

  def record_miss(self):
    self.details[Handler.CONTENT_TYPE_TITLE]['times'] += 1

  def describe(self):
    """
    解释到目前为止这个handler的执行情况
    """
    print('===================Topic Handler===================')
    # print('Content Type:', Handler.CONTENT_TYPE_TITLE)
    print('Content Type: 【%s】 ' % Handler.CONTENT_TYPE_TITLE)
    print('Total times:', self.details[Handler.CONTENT_TYPE_TITLE]['times'],end='\t\t')
    print('Found times:', self.details[Handler.CONTENT_TYPE_TITLE]['found_times'],end='\t\t')

    score_list = self.details[Handler.CONTENT_TYPE_TITLE]['rouge_scores']
    if len(score_list) == 0:
      rouge_score = -1
    else:
      rouge_score = sum(score_list) / len(score_list)
    print('Rouge-L avg score:,', rouge_score)
    # print('-------------------Topic Handler-------------------')
    print()
