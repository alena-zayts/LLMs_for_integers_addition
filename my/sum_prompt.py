SUM_PROMPT = '''
Q: What is 52 plus 148?

# solution in Python:


def solution():
    """What is 52 plus 148?"""
    a = 52
    b = 148
    plus = a + b
    result = plus
    return result





Q: What is -123456789 plus 4536181?

# solution in Python:


def solution():
    """What is -123456789 plus 4536181?"""
    a = -123456789
    b = 4536181
    plus = a + b
    result = plus
    return result





Q: What is 591 plus -103?

# solution in Python:


def solution():
    """What is 591 plus -103?"""
    a = 591
    b = -103
    plus = a + b
    result = plus
    return result





Q: What is -1 plus -20?

# solution in Python:


def solution():
    """What is -1 plus -20?"""
    a = -1
    b = -20
    plus = a + b
    result = plus
    return result





Q: What is 939492 plus 3929329?

# solution in Python:


def solution():
    """What is 939492 plus 3929329?"""
    a = 939492
    b = 3929329
    plus = a + b
    result = plus
    return result





Q: {question}

# solution in Python:
'''.strip() + '\n\n\n'