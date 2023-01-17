from glob import glob
import statistics
import re

collective = open("test_results_collective", 'w')
pivot = open("test_results_pivot", 'w')


averages = {}
for filename in glob("./test_results_ijcai/test/predict_if_present/*"):
    filename_splitted = filename.split("/")[-1].split("_")
    print(filename_splitted)
    if filename_splitted[1] == "test" and filename_splitted[0] == "results":
        lr = filename_splitted[2]
        modelname = filename_splitted[3]
        batch_size = filename_splitted[4]
        rep = filename_splitted[5]
        component = filename_splitted[6]
        f = open(filename, 'r')
        to_write = ""
        key = "{}_{}_{}_{}".format(component, modelname, lr, batch_size)
        for line in f:
                line_splitted = line.split(",")
                acc = line_splitted[0]
                precision = line_splitted[2]
                recall = line_splitted[3]
                f1_bin = line_splitted[1]

                if key not in averages:
                    averages[key] = [0.0, 0.0, 0.0, 0.0, 0, []]
                to_update = averages[key]
                to_update[0] += float(acc)
                to_update[1] += float(precision)
                to_update[2] += float(recall)
                to_update[3] += float(f1_bin.replace("\n", ""))
                to_update[4] += 1
                to_update[5].append(float(f1_bin.replace("\n", "")))

                averages[key] = to_update
                break

for k in averages:
    splitted = k.split("_")
    component = splitted[0]
    modelname = splitted[1]
    is_type_of_premise = len(splitted) == 5
    lr = splitted[2]
    values = averages[k]
    acc = values[0] / values [4]
    precision = values[1] / values[4]
    recall = values[2] / values[4]
    f1_minority = values[3] / values[4]
    if len(values[5]) > 1:
        stdev = statistics.stdev(values[5])
    else:
        stdev = 0
    to_write = "{},{},{},{},{},{},{}\n".format(modelname, lr, acc, precision, recall, f1_minority, stdev)
    if component == "Collective":
        collective.write(to_write)
    elif component == "pivot":
        pivot.write(to_write)

   
collective.close()
pivot.close()
