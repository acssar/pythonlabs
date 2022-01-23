import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', dest='inf', help="input file", default="matrix.txt")
parser.add_argument('-o', dest='out', help="output file", default="output.txt")
args = parser.parse_args()


def dim_check(n_x, m_x, n_y, m_y):
    if m_x > n_x:
        raise ValueError('The value of the kernel height exceeds the height of the matrix')
    if m_y > n_y:
        raise ValueError('The value of the kernel width exceeds the height of the matrix')


def read(file):
    """Function that reads an image and a filter from a file
       and counts rows and columns of each.

    Parameters
    ----------
    file : TextIO
        Input file.

        Format:  Image

                 Filter

        Example: 3 3 2 1 0
                 0 0 1 3 1
                 3 1 2 2 3
                 2 0 0 2 2
                 2 0 0 0 1

                 0 1 2
                 2 2 0
                 0 1 2

    Returns
    -------
    image : list
        Image of convolution.
    i_rows : int
        Number of rows in the image.
    i_cols : int
        Number of columns in the image.
    filter : list
        Filter of the convolution.
    f_rows : int
        Number of rows in the filter.
    f_cols : int
        Number of columns in the filter.

    """
    image = []
    filter = []
    i_rows = 0
    f_rows = 0

    for line in file:
        if line == '\n':
            break
        arr = line.replace('\n', '')
        arr = arr.split(' ')
        image.append(arr)
        i_rows += 1

    for line in file:
        arr = line.replace('\n', '')
        arr = arr.split(' ')
        filter.append(arr)
        f_rows += 1

    i_cols = len(image[0])
    f_cols = len(filter[0])
    return image, i_rows, i_cols, filter, f_rows, f_cols


def convolution(image, n_x, n_y, filter, m_x, m_y):
    """Function for calculating convolution

    Parameters
    ----------
    image : list
        Image of the convolution.
    n_x : int
        Number of rows in image.
    n_y : int
        Number of columns in image.
    filter : list
        Filter of the convolution.
    m_x : int
        Number of rows in filter.
    m_y : int
        Number of columns in filter.

    Returns
    -------
    c : list
        Result of convolution.

    """
    c = [[0] * (n_y - m_y + 1) for k in range(n_x - m_x + 1)]
    for i in range(n_x - m_x + 1):
        for j in range(n_y - m_y + 1):
            for u in range(m_x):
                for v in range(m_y):
                    c[i][j] += int(image[i + u][j + v]) * int(filter[u][v])
    return c


with open(args.inf, 'r') as infile:
    with open(args.out, "w") as outfile:
        img, img_rows, img_cols, fil, fil_rows, fil_cols = read(infile)
        dim_check(img_rows, fil_rows, img_cols, fil_cols)
        conv = convolution(img, img_rows, img_cols, fil, fil_rows, fil_cols)
        for row in conv:
            outfile.write(' '.join(list(map(str, row))))
            outfile.write('\n')
