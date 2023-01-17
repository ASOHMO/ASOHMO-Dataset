import glob
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
import sys

components = sys.argv[1:]

all_tweets = []
labels_justifications_dami = []
labels_justifications_jose = []

def labelComponents(text, justifications):
    if len(text.strip()) == 0:
        return []
    if len(justifications) == 0:
        return [0] * len(text.strip().split())

    if justifications[0] in text:
        parts = text.split(justifications[0])
        rec1 = labelComponents(parts[0], justifications[1:])
        rec2 = labelComponents(parts[1], justifications[1:])
        return rec1 + [1] * len(justifications[0].strip().split()) + rec2
    return [0] * len(text.strip().split())

filePatterns = ["./data/HateEval/agreement_tests/dami/*.ann", "./data/HateEval/agreement_tests/jose/*.ann"]

def labelComponentsFromAllExamples(filePattern, type_of_prem):
    quadrant_types_to_label = {"fact": 0, "value": 1, "policy": 2}
    all_labels = []
    for idxx, f in enumerate(glob.glob(filePattern)):
            name_of_premise = ""
            annotations = open(f, 'r')
            tweet = open(f.replace(".ann", ".txt"), 'r')
             # TODO: sacar todos los caracteres especiales
            tweet_text = tweet.read().replace("\n", "").replace("\t", "").replace(".", "").replace(",", "").replace("!", "").replace("#", "")
            type_of_quadrant = 0
            is_argumentative = True
            filesize = 0
            for idx, word in enumerate(annotations):
                filesize += 1
                ann = word.replace("\n", "").split("\t")
                if len(ann) > 1:
                    current_component = ann[1].lstrip()
                    if current_component.startswith("NonArgumentative"):
                       is_argumentative = False
                       break
                    if current_component.startswith(type_of_prem):
                        name_of_premise = ann[0]
                    elif current_component.split(" ")[0].startswith("QuadrantType"):
                        if name_of_premise == current_component.split(" ")[1]:
                            type_of_quadrant = quadrant_types_to_label[current_component.split(" ")[2].strip()]
                            break
            if filesize == 0 or not is_argumentative:
                type_of_quadrant = -1
            all_labels.append(type_of_quadrant)

    return all_labels

# TODO
#def labelsAgreementOverlap(annotator1, annotator2, percentage):
#    counting = False
#    for an1, an2 in zip(annotator1, annotator2):
#        if an1 == 1 or ann2 == 1:
#            if not counting:
#                counting = True

for component in components:

    dami_examples = labelComponentsFromAllExamples(filePatterns[0], component)
    jose_examples = labelComponentsFromAllExamples(filePatterns[1], component)

    only_argumentative_dami = [dami for dami, jose in zip(dami_examples, jose_examples) if dami != -1 and jose != -1]
    only_argumentative_jose = [jose for dami, jose in zip(dami_examples, jose_examples) if dami != -1 and jose != -1]

    print("Length of examples to compare (all the following numbers should be equal)")
    print("Agreement between annotator 1 and 3")
    print(cohen_kappa_score(only_argumentative_dami, only_argumentative_jose))
    print("F1 score between annotators 1 and 3")
    print(f1_score(dami_examples, jose_examples, average="macro"))