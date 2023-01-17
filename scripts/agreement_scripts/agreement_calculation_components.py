import glob
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
import sys

components = sys.argv[1:]

all_tweets = []
labels_justifications_dami = []
labels_justifications_jose = []

def labelComponents(text, component_text):
    if len(text.strip()) == 0:
        return []
    if len(component_text) == 0:
        return [0] * len(text.strip().split())

    if component_text[0] != "" and component_text[0] in text:
        parts = text.split(component_text[0])
        rec1 = labelComponents(parts[0], component_text[1:])
        rec2 = []
        if len(parts) > 2:
            rec2 = labelComponents(component_text[0].join(parts[1:]), component_text)
        else:
            rec2 = labelComponents(parts[1], component_text[1:])
        return rec1 + [1] * len(component_text[0].strip().split()) + rec2
    return [0] * len(text.strip().split())

filePatterns = ["./data/HateEval/agreement_tests/dami/*.ann", "./data/HateEval/agreement_tests/jose/*.ann"]

def labelComponentsFromAllExamples(filePattern, component):
    labels_per_example = []
    for idxx, f in enumerate(glob.glob(filePattern)):
        print("{}: {}".format(idxx, f))
        annotations = open(f, 'r')
        tweet = open(f.replace(".ann", ".txt"), 'r')
        # TODO: sacar todos los caracteres especiales
        tweet_text = tweet.read().replace("\n", "").replace("\t", "").replace(".", "").replace(",", "").replace("!", "").replace("#", "")
        component_text = []
        is_argumentative = True
        for idx, word in enumerate(annotations):
            ann = word.replace("\n", "").split("\t")
            #TODO: Si la justificacion esta dividida en mas de una parte esto no va a funcionar. Tampoco funciona si la anotacion corta una palabra a la mitad (por ejemplo, deja el punto final afuera)
            if ann[1].lstrip().startswith("NonArgumentative"):
                is_argumentative = False
                break
            if ann[1].lstrip().startswith(component):
                component_text.append(ann[2].lstrip().replace("\n","").replace("\t", "").replace(".", "").replace(",", "").replace("!", "").replace("#", ""))

        labels = [-1]
        if is_argumentative:
            labels = labelComponents(tweet_text, component_text)
        labels_per_example.append((f, labels))

    return labels_per_example

# TODO
#def labelsAgreementOverlap(annotator1, annotator2, percentage):
#    counting = False
#    for an1, an2 in zip(annotator1, annotator2):
#        if an1 == 1 or ann2 == 1:
#            if not counting:
#                counting = True



for component in components:

    print("Data for component " + component)
    dami_examples = labelComponentsFromAllExamples(filePatterns[0], component)
    jose_examples = labelComponentsFromAllExamples(filePatterns[1], component)

    only_argumentative_dami = [dami for dami, jose in zip(dami_examples, jose_examples) if dami[1][0] != -1 and jose[1][0] != -1]
    only_argumentative_jose = [jose for dami, jose in zip(dami_examples, jose_examples) if dami[1][0] != -1 and jose[1][0] != -1]

    idx = 1
    for dami, jose in zip(only_argumentative_dami, only_argumentative_jose):
        print("{}: {} - {}".format(idx, len(dami[1]), len(jose[1])))
        if len(dami[1]) != len(jose[1]):
            print(dami)
            print(jose)
        idx += 1

    dami_k = [l for label in only_argumentative_dami for l in label[1]]
    jose_k = [l for label in only_argumentative_jose for l in label[1]]

    print("Length of examples to compare (all the following numbers should be equal)")
    print(len(dami_k))
    print(len(jose_k))

    print("\n")
    print("Agreement between annotator 1 and 3")
    print(cohen_kappa_score(dami_k, jose_k))

    print("F1 score between annotators 1 and 3")
    print(f1_score(dami_k, jose_k))
