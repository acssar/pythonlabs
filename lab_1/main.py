import argparse

space = ' '


def current_row(m):
    row = []
    for i in range(m):
        if i == 0 or i == m - 1:
            row.append(1)
        else:
            c_row = current_row(m - 1)
            row.append(c_row[i - 1] + c_row[i])
    return row


def pascal_triangle(n):
    result = []
    for i in range(n):
        result.append(current_row(i + 1))

    number_width = len(str(max(result[-1]))) + 1

    for row in result:
        string = ''
        for number in row:
            number_string = str(number)
            string += number_string + ' ' * (number_width - len(number_string))
        print(string.center(number_width * n))


parser = argparse.ArgumentParser(description="Pascal triangle")

parser.add_argument('-n', dest="N", default=3, type=int)

args = parser.parse_args()

pascal_triangle(args.N)
