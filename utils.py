def prepare_file(source_path, target_path):
    f_source = open(source_path, 'r')
    f_target = open(target_path, 'w+')
    counter = 0
    for line in f_source:
        l = line
        if (counter > 0 and line.strip()):
            l = str(counter) + ',' + line
        f_target.write(l)
        counter += 1
    f_source.close()
    f_target.close()

def print_results(text):
    print(text)
    path = 'results/results.txt'
    f = open(path, 'a')
    f.write(text)
    f.close()