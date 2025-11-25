def main(file):
    with open(file, 'r') as f:
        content = f.readlines()
    
    # 1. Strip whitespace characters like ` ` at the end of each line
    content = [
        line.strip() for line in content
    ]

    # 2. Remove lines starting with '#' and empty lines
    content = [
        line for line in content if line and not line.startswith('#')
    ]

    # 3. Join the lines back into a single string (with ' ' as separator)
    result = ' '.join(content)

    # 4. Finally, print the resulting string
    print(result)

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
