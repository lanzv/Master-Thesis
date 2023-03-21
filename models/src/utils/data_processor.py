import re


def round_md_file(input_file_path="./input.MD", output_file_path="./output.MD"):
    f = open(input_file_path, "r")
    content = f.read()
    new_content = content[ : : -1 ][ : : -1 ]
    pattern = re.compile(r'(\| *)([0-9]+\.[0-9]+)( *\|)')

    for a, number, b in re.findall(pattern, content):
        new_number = "|{:.2f}%|".format(float(number)*100)
        new_content = new_content.replace(a+number+b, new_number)

    new_content2 = new_content[ : : -1 ][ : : -1 ]
    pattern = re.compile(r'(\| *)([0-9]+\.[0-9]+)( *\|)')

    for a, number, b in re.findall(pattern, new_content):
        new_number = "|{:.2f}%|".format(float(number)*100)
        new_content2 = new_content2.replace(a+number+b, new_number)

    content = new_content2.replace("%", "")

    f = open(output_file_path, "w")
    f.write(content)
    f.close()