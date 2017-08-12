import random

train_size = 0.6
valid_size = 0.2
test_size = 1 - valid_size - train_size


if __name__ == '__main__':

    lines = []

    with open('multinli_0.9_train.txt', 'r', encoding='utf-8', errors='ignore') as file:
        for line in file:
            lines.append(line)

    lines = lines[1:]

    random.shuffle(lines)

    size = len(lines)

    train_line_num = int(size * train_size)
    valid_line_num = int(size * valid_size)
    test_line_num = int(size * test_size)

    train_lines = lines[:train_line_num]
    valid_lines = lines[train_line_num + 1:train_line_num+valid_line_num]
    test_lines = lines[-test_line_num:]

    with open('train.txt', 'w', encoding='utf-8') as file:
        file.write(''.join(train_lines))

    with open('valid.txt', 'w', encoding='utf-8') as file:
        file.write(''.join(valid_lines))

    with open('test.txt', 'w', encoding='utf-8') as file:
        file.write(''.join(test_lines))
