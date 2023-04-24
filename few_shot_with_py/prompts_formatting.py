def code_for_one_task(a, b):
    return f'''
    def solution():
        """What is {a} plus {b}?"""
        a = {a}
        b = {b}
        plus = a + b
        result = plus
        return result
        
        
        
        
        
        
    '''

def header_for_one_task(a, b):
    return f'''
    Q: What is {a} plus {b}?

    # solution in Python:
    '''

def full_question_with_code_for_one_task(a, b):
    return header_for_one_task(a, b) + code_for_one_task(a, b)

examples = [
    [121231, 349340],
    [-12, 31323],
    [93, -2901201],
    [-123, -132],
    [239239876568, 876578290323],
]


def generate_prompt():
    prompt = ''
    for example in examples:
        prompt += full_question_with_code_for_one_task(*example)

    return prompt + \
'''
Q: What is {a} plus {b}?

# solution in Python:
'''.strip() + '\n\n\n'


