from glob import glob
import statistics
import re

types_both = open("test_results_types", "w")
type_justification = open("test_results_type_justification", "w")
type_conclusion = open("test_results_type_conclusion", "w")


averages = {}
for filename in glob("./test_results_ijcai/test/premise_types/*"):
    filename_splitted = filename.split("/")[-1].split("_")
    print(filename_splitted)
    if filename_splitted[1] == "test" and filename_splitted[0] == "results":
        lr = filename_splitted[2]
        modelname = filename_splitted[3]
        batch_size = filename_splitted[4]
        rep = filename_splitted[5]
        component = filename_splitted[7]
        f = open(filename, 'r')
        to_write = ""
        key = "{}_{}_{}_{}".format(component, modelname, lr, batch_size)
        for line in f:
                line_splitted = line.split(",")
                acc = line_splitted[0]
                precision = line_splitted[2]
                recall = line_splitted[3]
                f1_bin = line_splitted[1]
                per_class_f1 = re.split( r' +', line_splitted[4].replace("\n","").replace("[", "").replace("]", ""))

                if key not in averages:
                    averages[key] = [0.0, 0.0, 0.0, 0.0, 0, [], 0.0, 0.0, 0.0]
                to_update = averages[key]
                to_update[0] += float(acc)
                to_update[1] += float(precision)
                to_update[2] += float(recall)
                to_update[3] += float(f1_bin.replace("\n", ""))
                to_update[4] += 1
                to_update[5].append(float(f1_bin.replace("\n", "")))
                to_update[6] += float(per_class_f1[0])
                to_update[7] += float(per_class_f1[1])
                to_update[8] += float(per_class_f1[2])

                averages[key] = to_update
                break

for k in averages:
    splitted = k.split("_")
    component = splitted[0]
    modelname = splitted[1]
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
    f1_fact = values[6] / values[4]
    f1_value = values[7] / values[4]
    f1_policy = values[8] / values[4]
    to_write = "{},{},{},{},{},{},{},{},{},{}\n".format(modelname, lr, acc, precision, recall, f1_minority, stdev, f1_fact, f1_value, f1_policy)
    if component == "joint-premises":
        types_both.write(to_write)
    elif component == "tested-with-conc":
        type_conclusion.write(to_write)
    elif component == "tested-with-just":
        type_justification.write(to_write)

   
types_both.close()
type_conclusion.close()
type_justification.close()
