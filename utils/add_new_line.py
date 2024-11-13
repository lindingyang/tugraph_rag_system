input_file = '../python_api'
output_file = 'output.txt'

with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

with open(output_file, 'w', encoding='utf-8') as f:
    for line in lines:
        f.write(line.rstrip() + '\n\n')
