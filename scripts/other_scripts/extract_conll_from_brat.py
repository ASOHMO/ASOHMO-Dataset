from glob import glob
from pysentimiento import preprocessing
import glob
import re
import random
import argparse
import os

parser = argparse.ArgumentParser(description="Train models for identifying argumentative components inside the ASFOCONG dataset")
parser.add_argument('--create_train_test_partitions', type=bool, default=False, help="Separate tweets in train dev and test partitions")


args = parser.parse_args()
create_train_test_partitions = args.create_train_test_partitions

COMPONENTS = ["Property", "Collective", "Premise1Conclusion", "Premise2Justification", "pivot"]

def replaceSpace(string):
    pattern = " " + '{2,}'
    string = re.sub(pattern, " ", string)
    return string

def labelComponents(text, component_text, component, assert_check=False):

    if assert_check:
        for cmpnt in component_text:
            assert(cmpnt in text)

    if len(text.strip()) == 0:
        return []
    if len(component_text) == 0:
        return ["O"] * len(text.strip().split(" "))

    if component_text[0] != "" and component_text[0] in text:
        parts = text.split(component_text[0])
        rec1 = labelComponents(parts[0].strip(), component_text[1:], component)
        rec2 = []
        if len(parts) > 2:
            rec2 = labelComponents(component_text[0].join(parts[1:]), component_text, component)
        else:
            rec2 = labelComponents(parts[1], component_text[1:], component)

        return rec1 + ([component] * len(component_text[0].strip().split(" "))) + rec2
    return ["O"] * len(text.strip().split())

def normalize_text(tweet_text, arg_components_text):
    for splitter in arg_components_text:

        if splitter == tweet_text:
            return tweet_text.split(" ")
        if len(splitter.replace(" ", "")) > 0 and splitter.replace(" ", "") in tweet_text:
            tweet_text = replaceSpace(tweet_text.replace(splitter.replace(" ", ""), " " + splitter + " "))
                
        if splitter not in tweet_text and splitter.lower() in tweet_text:
            splitter = splitter.lower()
        assert (splitter in tweet_text)

        tweet_text = replaceSpace((" " + splitter + " ").join(tweet_text.split(splitter))).strip()
    return tweet_text.split(" ")

def delete_unwanted_chars(text):
    if re.match("[a-zA-Z]+#", text):
        text = text.replace("#", " #")
    text = " #".join(text.split("#"))
    return replaceSpace(text.lower().replace("\n", "").replace("\t", " ").replace(".", " ").replace(",", " ").replace("!", "").replace('“', '"').replace('”', '"').replace('…', '').replace("’", "").replace("–", " ").replace("‘", "").replace("—", " ").replace("·", " ").replace(";", " ").replace("'", ""))


def labelComponentsFromAllExamples(filePatterns, prefix=""):
    count_cn = {}
    count_argumentative = 0
    for f in filePatterns:
            annotations = open(f, 'r')
            tweet = open(f.replace(".ann", ".txt"), 'r')
            tweet_text = delete_unwanted_chars(tweet.read())
            preprocessed_text = preprocessing.preprocess_tweet(tweet_text, lang='en', user_token="@user", url_token="link", hashtag_token="hashtag")
            component_texts = {}
            is_argumentative = True
            filesize = 0
            name_of_premises = {}
            type_of_premises = {}
            cn_references = {}
            counter_narratives = {}
            for idx, word in enumerate(annotations):
                filesize += 1
                ann = word.replace("\n", "").split("\t")
                if len(ann) > 1:
                    current_component = ann[1].lstrip()
                    if current_component.startswith("NonArgumentative"):
                        is_argumentative = False
                        count_argumentative += 1
                        break
                    if current_component.startswith("Premise"):
                        name_of_premises[ann[0]] = current_component.split(" ")[0]
                    if current_component.startswith("QuadrantType"):
                        info_splitted = current_component.split(" ")
                        type_of_premises[name_of_premises[info_splitted[1]]] = info_splitted[2]

                    if current_component.startswith("Property") or current_component.startswith("Collective") or current_component.startswith("pivot") or current_component.startswith("Premise1Conclusion") or current_component.startswith("Premise2Justification"):
                        new_component = replaceSpace(preprocessing.preprocess_tweet(delete_unwanted_chars(ann[2].lstrip()), lang='en', user_token="@user", url_token="link", hashtag_token="hashtag"))
                        for cmpnt in COMPONENTS:
                            new_component_list_aux = []
                            if cmpnt not in component_texts:
                                component_texts[cmpnt] = []
                            for component in component_texts[cmpnt]:

                                if new_component in preprocessed_text:
                                    preprocessed_text = " ".join(normalize_text(preprocessed_text, [new_component]))

                                if component in new_component or (component.replace(" ","") in new_component):
                                    new_component = " ".join(normalize_text(new_component, [component]))
                                if new_component in component or new_component.replace(" ", "") in component:
                                    new_component_list_aux.append(" ".join(normalize_text(component, [new_component])))
                                else:
                                    new_component_list_aux.append(component)
                            
                            component_texts[cmpnt] = new_component_list_aux

                            if current_component.startswith(cmpnt):
                                component_texts[cmpnt].append(new_component)

                    if current_component.startswith("CounterNarrative"):
                        cn_references[ann[0]] = current_component.split(" ")[0]
                    if current_component.startswith("AnnotatorNotes"):
                        code = current_component.split(" ")[1]
                        print(f)
                        counter_narratives[cn_references[code]] = ann[2]
                        if cn_references[code] not in count_cn:
                            count_cn[cn_references[code]] = []
                        count_cn[cn_references[code]].append(ann[2])

            components_list = [compnent for key in component_texts for compnent in component_texts[key]]
            normalized_text = normalize_text(preprocessed_text, components_list)
            assert(not (is_argumentative and ("Premise1Conclusion" not in type_of_premises or "Premise2Justification" not in type_of_premises)))

            component_labels = []
            argumentative = ["O" if is_argumentative else "NoArgumentative"] * len(normalized_text)
            component_labels.append(argumentative)
            for cmpnt in ["Premise2Justification", "Premise1Conclusion", "Collective", "Property", "pivot"]:
                if not is_argumentative:
                    labels = ["O"] * len(normalized_text)
                    type_of_justification = ["O"] * len(normalized_text)
                    type_of_conclusion = ["O"] * len(normalized_text)
                else:

                    if not cmpnt in component_texts:
                        labels = ["O"] * len(normalized_text)
                    else:
                        labels = labelComponents(" ".join(normalized_text), component_texts[cmpnt], cmpnt, assert_check=True)
                    if cmpnt == "Premise2Justification":
                        type_of_justification = [type_of_premises[lbl] if lbl == "Premise2Justification" else "O" for lbl in labels]
                    elif cmpnt == "Premise1Conclusion":
                        type_of_conclusion = [type_of_premises[lbl] if lbl == "Premise1Conclusion" else "O" for lbl in labels]

                assert(len(normalized_text) == len(labels))
                component_labels.append(labels)


            component_labels.append(type_of_justification)
            component_labels.append(type_of_conclusion)

            conll = open(prefix + f.split("/")[-1].replace(".ann", ".conll"), "w")
            for idx, wrd in enumerate(normalized_text):
                line = [wrd]
                for i in range(len(component_labels)):
                    line.append(component_labels[i][idx])
                jointed = "\t".join(line)
                conll.write("{}\n".format(jointed))
            conll.close()

            cn = open(prefix + f.split("/")[-1].replace(".ann", ".cn"), "w")
            for type_of_cn in sorted(counter_narratives.keys()):
                cn.write("{}\n".format(counter_narratives[type_of_cn]))
            cn.close()
    print("CN STATISTICS")
    for cn_typ in count_cn:
        print("{}: {}".format(cn_typ, len(count_cn[cn_typ])))
    print("Non Argumentative:")
    print(count_argumentative)


filePatterns = ["./data/HateEval/partition_{}/hate_tweet_*.ann".format(num) for num in range(1,11)]

allFiles = []
for pattern in filePatterns:
    for f in glob.glob(pattern):
        allFiles.append(f)

if not create_train_test_partitions:
    labelComponentsFromAllExamples(allFiles)
else:
    allFilesCp = allFiles.copy()
    random.Random(41).shuffle(allFilesCp)
    os.mkdir("train")
    os.mkdir("test")
    os.mkdir("dev")
    labelComponentsFromAllExamples(allFilesCp[:770], prefix="train/")
    labelComponentsFromAllExamples(allFilesCp[770:870], prefix="dev/")
    labelComponentsFromAllExamples(allFilesCp[870:], prefix="test/")