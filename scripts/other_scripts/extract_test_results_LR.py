from glob import glob
import statistics

collective = open("test_results_collective", 'w')
proper = open("test_results_property", 'w')
pivot = open("test_results_pivot", 'w')
conclusion = open("test_results_conclusion", 'w')
justification = open("test_results_justification", 'w')
argumentative = open("test_results_argumentative", 'w')
type_justification = open("test_results_type_justification", "w")
type_conclusion = open("test_results_type_conclusion", "w")


averages = {}
for filename in glob("./final_version_test_results_LR_no_embed/*"):
    filename_splitted = filename.split("/")[2].split("_")
    print(filename_splitted)
    if filename_splitted[1] == "test":
        lr = "C-{}".format(filename_splitted[2])
        modelname = filename_splitted[3]
        component = filename_splitted[6]
        rep = filename_splitted[5]
        f = open(filename, 'r')
        to_write = ""
        key = "{}_{}_{}".format(component, modelname, lr)
        if component == "type-of-premise":
            key = "{}_{}_{}_{}".format(component, modelname, lr, filename_splitted[7])
        for line in f:
                line_splitted = line.split(",")
                acc = line_splitted[0]
                precision = line_splitted[1]
                recall = line_splitted[2]
                f1_minority = line_splitted[3]

                if key not in averages:
                    averages[key] = [0.0, 0.0, 0.0, 0.0, 0, []]
                to_update = averages[key]
                to_update[0] += float(acc)
                to_update[1] += float(precision)
                to_update[2] += float(recall)
                to_update[3] += float(f1_minority.replace("\n", ""))
                to_update[4] += 1
                to_update[5].append(float(f1_minority.replace("\n", "")))
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
    print(component)
    print(lr)
    print(values[5])
    stdev = statistics.stdev(values[5])
    to_write = "{},{},{},{},{},{},{}\n".format(modelname, lr, acc, precision, recall, f1_minority, stdev)
    if component == "Collective":
        collective.write(to_write)
    elif component == "Property":
        proper.write(to_write)
    elif component == "pivot":
        pivot.write(to_write)
    elif component == "Premise1Conclusion":
        conclusion.write(to_write)
    elif component == "Premise2Justification":
        justification.write(to_write)
    elif component == "NonArgumentative":
        argumentative.write(to_write)
    elif component == "type-of-premise":
        if splitted[3] == "Premise1Conclusion":
            type_conclusion.write(to_write)
        elif splitted[3] == "Premise2Justification":
            type_justification.write(to_write)
   
collective.close()
proper.close()
pivot.close()
conclusion.close()
justification.close()
argumentative.close() 
