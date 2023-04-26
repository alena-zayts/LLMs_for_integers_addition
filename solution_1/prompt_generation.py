CODE_START_MARKER = '# solution in Python:'
QUESTION_START_MARKER = 'Q:'
FUNCTION_CALL = 'solution()'

examples_numbers = [
    [121231, 349340],
    # [-12, 31323],
    # [93, -2901201],
    # [-123, -132],
    # [239239876568, 876578290323],
]

EXAMPLES_AMOUNT = len(examples_numbers)


def code_for_one_task(a, b):
    return f'''
    def {FUNCTION_CALL}:
        return {a} + {b}

    '''


def header_for_one_task(a, b):
    return f'''
    {QUESTION_START_MARKER} What is {a} plus {b}?

    {CODE_START_MARKER}
    '''


def full_question_with_code_for_one_task(a, b):
    return header_for_one_task(a, b) + code_for_one_task(a, b)


def generate_prompt():
    prompt = ''
    for example in examples_numbers:
        prompt += full_question_with_code_for_one_task(*example)

    return prompt + header_for_one_task('{a}', '{b}')


def count_code_answer_length(a, b):
    return len(code_for_one_task(a, b))


