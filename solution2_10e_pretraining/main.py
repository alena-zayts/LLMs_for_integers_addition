import argparse
from solver import Solver2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evalute T5 on arithmetic problems.')
    parser.add_argument('--a', type=int, required=True, help='first integer summand')
    parser.add_argument('--b', type=int, required=True, help='second integer summand')

    args = parser.parse_args()

    try:
        solver = Solver2()
        a = int(args.a)
        b = int(args.b)
        res_int, meta_info = solver.calc_sum(a, b)
        print(f'{args.a} + {args.b} = {res_int}')

    except Exception as e:
        print(e)
