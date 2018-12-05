
def main(): 
    with open("hle.out") as f:
        lines = f.readlines()

    for line in lines:
        newline = []
        while "?" not in line:
            line.replace('[','(')
            line.replace(']',')')
            newline.append(line)
        break
if __name__ == '__main__':
    main()
