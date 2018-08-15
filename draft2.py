import json

with open(r'C:\Users\hasee\Documents\Tencent Files\260174969\FileRecv\dev.predicted.json', encoding='utf-8') as f:
  data = f.readline().strip()
  data = json.loads(data)

import pandas as pd
from GLOBAL_CONFIG import *
from testing.rouge import RougeL

questions = pd.read_csv(question_csv_path)
paragraphs = pd.read_csv(paragraphs_csv_path)

rogue_eval = RougeL()

# for
total_count = 0
target_count = 0
for d in data[:100]:
  article_id = int(d['article_id'])
  cur_questions = questions.loc[questions['article_id'] == article_id]

  q_dict = {}
  print(article_id)
  for idx in cur_questions.index:
    q_id, a_id, question_type, question, ans = cur_questions.loc[idx].values
    q_dict[q_id] = (question, ans)

  qs = d['questions']
  for q in qs:
    total_count += 1
    qid = q['question_id']
    pred_ans = q['answer']

    question, ans = q_dict[qid]
    score = rogue_eval.calc_score(pred_ans, ans)
    if score <= 0.6:
      target_count += 1
      print(score, question, 'gt:[%s]' % ans, 'pred:[%s]' % pred_ans)
  print('-' * 80)
print(total_count, target_count)
