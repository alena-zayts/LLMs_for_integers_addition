def translate_task(a_int: int, b_int: int):
    result_int = a_int + b_int

    a_str = convert_to_10ebased(str(a_int))
    b_str = convert_to_10ebased(str(b_int))
    result_str = convert_to_10ebased(str(result_int))

    question = f'What is {a_str} plus {b_str}?'
    return {
        'a_int': a_int,
        'b_int': b_int,
        'expected_result_int': result_int,

        'a_str': a_str,
        'b_str': b_str,
        'expected_result_str': result_str,

        'question': question,
    }

def compute_exact_match(predicted_answer, correct_answer) -> bool:
    predicted_answer = predicted_answer.strip().lower()
    correct_answer = correct_answer.strip().lower()
    return predicted_answer == correct_answer


def convert_to_10ebased(number: str) -> str:
    signal = None
    if number[0] == '-':
        signal = '-'
        number = number[1:]

    output = []
    for i, digit in enumerate(number[::-1]):
        output.append('10e' + str(i))
        output.append(digit)

    if signal:
        output.append(signal)

    # as we want it to _not_ be inverted, then we invert it.
    output = output[::-1]

    return ' '.join(output)

def convert_from_10ebased(number: str) -> str:
    signal = ''
    if number[0] == '-':
        signal = '-'
        number = number[2:]

    split_number = number.split()
    assert len(split_number) % 2 == 0

    max_exp = int(split_number[1][3:])
    answer_length = max_exp + 1
    n_non_zeroes = len(split_number) // 2

    output = ['0'] * answer_length

    for i in range(n_non_zeroes):
        base = split_number[2 * i]
        exp = int(split_number[2 * i + 1][3:])
        output[exp] = base

    output.reverse()
    res = signal + ''.join(output)

    return res


if __name__ == '__main__':
    x = str(-134)
    x_10 = convert_to_10ebased(x)
    x_back = convert_from_10ebased(x_10)
    print(x)
    print(x_10)
    print(x_back)

    for i in range(-10001, 10002, 1):
        print(i)
        x = str(i)
        x_10 = convert_to_10ebased(x)
        x_back = convert_from_10ebased(x_10)
        assert int(x) == int(x_back)

