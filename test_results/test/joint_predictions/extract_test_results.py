from glob import glob
import statistics
import re

collective_property = open("test_results_collective-property", 'w')
premises = open("test_results_premises", 'w')


averages = {}
for filename in glob("./test_results_ijcai/test/joint_predictions/complete_info/*"):
    filename_splitted = filename.split("/")[-1].split("_")
    print(filename_splitted)
    if filename_splitted[1] == "test" and filename_splitted[0] == 'results':
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
                per_class_f1 = re.split( r' +', line_splitted[4].replace("\n","").replace("[", "").replace("]", ""))
                per_class_precision = re.split( r' +', line_splitted[5].replace("\n","").replace("[", "").replace("]", ""))
                per_class_recall = re.split( r' +', line_splitted[6].replace("\n","").replace("[", "").replace("]", ""))

                if key not in averages:
                    averages[key] = [0.0, 0.0, 0.0, 0.0, 0, [], 0.0, [], 0.0, [], 0.0, [], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                to_update = averages[key]
                to_update[0] += float(acc)
                to_update[1] += float(precision)
                to_update[2] += float(recall)
                to_update[3] += float(f1_bin.replace("\n", ""))
                to_update[4] += 1
                to_update[5].append(float(f1_bin.replace("\n", "")))
                to_update[6] += float(per_class_f1[0])
                to_update[7].append(float(per_class_f1[0]))
                to_update[8] += float(per_class_f1[1])
                to_update[9].append(float(per_class_f1[1]))
                to_update[10] += float(per_class_f1[2])
                to_update[11].append(float(per_class_f1[2]))
                to_update[12] += float(per_class_precision[0])
                to_update[13] += float(per_class_precision[1])
                to_update[14] += float(per_class_precision[2])
                to_update[15] += float(per_class_recall[0])
                to_update[16] += float(per_class_recall[1])
                to_update[17] += float(per_class_recall[2])

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
    if len(values[7]) > 1:
        stdev_fact = statistics.stdev(values[7])
    else:
        stdev_fact = 0
    f1_value = values[8] / values[4]
    if len(values[9]) > 1:
        stdev_value = statistics.stdev(values[9])
    else:
        stdev_value = 0
    f1_policy = values[10] / values[4]
    if len(values[11]) > 1:
        stdev_policy = statistics.stdev(values[11])
    else:
        stdev_policy = 0
    precision_fact = values[12] / values[4]
    precision_value = values[13] / values[4]
    precision_policy = values[14] / values[4]
    recall_fact = values[15] / values[4]
    recall_value = values[16] / values[4]
    recall_policy = values[17] / values[4]
    to_write = "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(modelname, lr, acc, precision, recall, f1_minority, stdev, f1_fact, stdev_fact, f1_value, stdev_value, f1_policy, stdev_policy, precision_fact, precision_value, precision_policy, recall_fact, recall_value, recall_policy)
    if component == "Collective-Property":
        collective_property.write(to_write)
    elif component == "Premises":
        premises.write(to_write)

   
collective_property.close()
premises.close()