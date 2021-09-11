import argparse


def current_row(m, prev):
    row = []
    for i in range(m):
        if i == 0 or i == m - 1:
            row.append(1)
        else:
            row.append(prev[i - 1] + prev[i])
    return row


def pascal_triangle(n):
    result = []
    row = []
    for i in range(n):
        row = current_row(i+1, row)
        result.append(row)

    number_width = len(str(max(result[-1]))) + 1

    for row in result:
        string = ''
        for number in row:
            number_string = str(number)
            string += number_string + ' ' * (number_width - len(number_string))
        print(string.center(number_width * n))


parser = argparse.ArgumentParser(description="Pascal triangle")

parser.add_argument('-n', dest="N", default=5, type=int)  # TODO: help

args = parser.parse_args()

pascal_triangle(args.N)
