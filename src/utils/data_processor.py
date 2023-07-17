import re


def round_md_file(input_file_path="./input.MD", output_file_path="./output.MD"):
    """
    Process the MarkDown file and change all values in tables and round them on 2 decimal places.
    First, values are multplied by 100 and the suffix '%' is added. Second, the round function is applied.
    Save the result to new markdown file.

    Parameters
    ----------
    input_file_path : str
        path to markdown file of not rounded numbers
    output_file_path : str
        path to markdown file that will be copy of the input markdown file but with rounded percentage values
    """
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