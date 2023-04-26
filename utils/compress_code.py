import re
import sys

def compress_code(code: str) -> str:
    lines = code.split('\n')
    compressed_lines = []

    for line in lines:
        line = line.strip()  # Remove leading and trailing whitespace
        line = re.sub(r'\s*([=+\-*/(,)])\s*', r'\1', line)  # Remove spaces around operators and brackets
        compressed_lines.append(line)

    compressed_code = ' '.join(compressed_lines)
    return compressed_code

def compress_file(filename: str) -> str:
    with open(filename, 'r') as file:
        code = file.read()
    compressed_code = compress_code(code)
    return compressed_code

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compress_script.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    compressed = compress_file(filename)
    print(compressed)
