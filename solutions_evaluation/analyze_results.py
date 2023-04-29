import json
import pandas as pd

pd.set_option('display.expand_frame_repr', False)

results = list(map(json.loads, open('test_examples_solution1_results.jsonl')))
data = pd.DataFrame(results)

data['a_neg'] = (data['a'] < 0).astype(int)
data['b_neg'] = (data['b'] < 0).astype(int)
data['answer_neg'] = (pd.to_numeric(data['answer_int']) < 0).astype(int)
data['target_neg'] = (data['target'] < 0).astype(int)
data['score'] = (data['answer_int'] == data['target']).astype(int)
data['answer_was_given'] = (data['answer_int'] != '').astype(int)

# data = data.drop(['target', 'a', 'b', 'answer_int', 'meta_info'], axis=1)


# print(data)
# print(data.columns)

print('All errors')
print(data[data['score'] == 0])

data[data['score'] == 0].to_excel('solution1_fails.xlsx')

# print('When signs are different')
# print(data[data['answer_neg'] != data['target_neg']])

print(f'Accuracy: {data["score"].mean()}')
