# coding = utf-8
"""
回答可以直接用模板匹配解决的问题
"""
from feature_handler.templete_common import *
from feature_handler.handler import Handler
from utils.rouge import RougeL
import re


def _get_rouge_avg_socre(scores):
  if len(scores) == 0:
    rouge_score = -1
  else:
    rouge_score = sum(scores) / len(scores)
  return rouge_score


class BasicTempleteHandler(object):
  def __init__(self, feature_words_path, stop_words_path, is_train):
    self.name = 'Basic Templete Handler'
    with open(feature_words_path, encoding='utf-8') as f:
      self.feature_words = [line.strip() for line in f.readlines()]
    with open(stop_words_path, encoding='utf-8') as f:
      self.stop_words = [line.strip() for line in f.readlines()]

    self.details = {
      'times': 0
    }
    self.details[Handler.CONTENT_TYPE_TITLE] = {
      'found_times': 0,
    }
    self.details[Handler.CONTENT_TYPE_ARTICLE] = {
      'found_times': 0,
    }
    if is_train:
      self.details[Handler.CONTENT_TYPE_TITLE]['rouge_scores'] = []
      self.details[Handler.CONTENT_TYPE_ARTICLE]['rouge_scores'] = []

      self.rogue_eval = RougeL()

  def ans_question(self, title, article, question,
                   question_ans=None):
    self.details['times'] += 1

    templetes = identify_templete(question, self.feature_words, self.stop_words)
    for templete in templetes:

      found_ans = match_content(templete, title, self.feature_words)
      if found_ans is not None:
        if question_ans is not None:
          self.record_found(Handler.CONTENT_TYPE_TITLE, found_ans, question_ans)
        return found_ans

      found_ans = match_content(templete, article, self.feature_words)
      if found_ans is not None:
        if question_ans is not None:
          self.record_found(Handler.CONTENT_TYPE_ARTICLE, found_ans, question_ans)
        return found_ans

    return None

  def record_found(self, content_type, pred_ans, gt_ans):
    self.details[content_type]['found_times'] += 1
    score = self.rogue_eval.calc_score(pred_ans, gt_ans)
    self.details[content_type]['rouge_scores'].append(score)

  def describe(self):
    title_score_list = self.details[Handler.CONTENT_TYPE_TITLE]['rouge_scores']
    article_score_list = self.details[Handler.CONTENT_TYPE_ARTICLE]['rouge_scores']

    print('===================%s===================' % self.name)
    print('Total times:', self.details['times'], end='\t\t')
    print('Total found times:',
          self.details[Handler.CONTENT_TYPE_TITLE]['found_times'] +
          self.details[Handler.CONTENT_TYPE_ARTICLE]['found_times'], end='\t\t')
    print('Total Rouge-L avg score:', _get_rouge_avg_socre(title_score_list + article_score_list))

    print('Content Type: 【%s】 ' % Handler.CONTENT_TYPE_TITLE)
    print('Found times:', self.details[Handler.CONTENT_TYPE_TITLE]['found_times'], end='\t\t')
    print('Rouge-L avg score:,', _get_rouge_avg_socre(title_score_list))

    print('Content Type: 【%s】 ' % Handler.CONTENT_TYPE_ARTICLE)
    print('Found times:', self.details[Handler.CONTENT_TYPE_ARTICLE]['found_times'], end='\t\t')
    print('Rouge-L avg score:,', _get_rouge_avg_socre(article_score_list))

    print()


class ShrinkTempleteHandler(BasicTempleteHandler):
  def __init__(self, feature_words_path, stop_words_path, is_train):
    super().__init__(feature_words_path, stop_words_path, is_train)
    self.name = 'Shrink Templete Handler'

  def ans_question(self, title, article, question,
                   question_ans=None):
    self.details['times'] += 1

    templetes = identify_templete(question, self.feature_words, self.stop_words)
    templetes = templete_shrink(templetes)
    for templete in templetes:

      found_ans = match_content(templete, title, self.feature_words)
      if found_ans is not None:
        if question_ans is not None:
          self.record_found(Handler.CONTENT_TYPE_TITLE, found_ans, question_ans)
        return found_ans

      found_ans = match_content(templete, article, self.feature_words)
      if found_ans is not None:
        if question_ans is not None:
          self.record_found(Handler.CONTENT_TYPE_ARTICLE, found_ans, question_ans)
        return found_ans

    return None


class FuzzyTempleteHandler(BasicTempleteHandler):
  def __init__(self, feature_words_path, stop_words_path, is_train):
    super().__init__(feature_words_path, stop_words_path, is_train)
    self.name = 'Fuzzy Templete Handler'

  def ans_question(self, title, article, question,
                   question_ans=None):
    self.details['times'] += 1
    templetes = identify_templete(question, self.feature_words, self.stop_words, use_templete_stop_words=True)
    templetes = templete_shrink(templetes)
    for templete in templetes:

      found_ans = match_content(templete, title, self.feature_words)
      if found_ans is not None:
        if question_ans is not None:
          self.record_found(Handler.CONTENT_TYPE_TITLE, found_ans, question_ans)
        return found_ans

      found_ans = match_content(templete, article, self.feature_words)
      if found_ans is not None:
        if question_ans is not None:
          self.record_found(Handler.CONTENT_TYPE_ARTICLE, found_ans, question_ans)
        return found_ans

    return None
