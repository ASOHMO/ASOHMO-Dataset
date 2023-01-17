import glob
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
import sys

NUMBER_OF_PARTITIONS = 10

components = sys.argv[1:]

all_tweets = []
labels_justifications_dami = []
labels_justifications_jose = []


#filePatterns = ["./data/HateEval/partition_{}/hate_tweet_*.ann".format(partition_num) for partition_num in range(1, NUMBER_OF_PARTITIONS + 1)]

filePatterns = ["./data/HateEval/partition_spanish/hate_tweet_*.ann"]

def delete_unwanted_chars(text):
        return text.replace("\n", "").replace("\t", "").replace(".", "").replace(",", "").replace("!", "").replace("#", "").replace('“', '"').replace('”', '"').replace('…', '').replace("’", "").replace("–", " ").replace("‘", "").replace("—", "").replace("·", "")

def labelComponentsFromAllExamples(filePattern, component):
    non_argumentatives = 0
    have_component = 0
    component_words = 0
    all_words = 0
    type_fact = 0
    type_value = 0
    type_policy = 0
    cna = 0
    cnb = 0
    cnc = 0
    cnf = 0
    for idxx, f in enumerate(glob.glob(filePattern)):
        print("{}: {}".format(idxx, f))
        annotations = open(f, 'r')
        tweet = open(f.replace(".ann", ".txt"), 'r')
        # TODO: sacar todos los caracteres especiales
        tweet_text = delete_unwanted_chars(tweet.read())
        all_words += len(tweet_text.split())
        component_text = []
        is_argumentative = True
        premise_codes = []
        fact = False
        value = False
        policy = False
        for idx, word in enumerate(annotations):
            ann = word.replace("\n", "").split("\t")
            #TODO: Si la justificacion esta dividida en mas de una parte esto no va a funcionar. Tampoco funciona si la anotacion corta una palabra a la mitad (por ejemplo, deja el punto final afuera)
            if len(ann) > 1:
               current_component = ann[1].lstrip()
               if current_component.startswith("NonArgumentative"):
                   is_argumentative = False
                   break
               if current_component.startswith("CounterNarrativeA"):
                   cna += 1
               if current_component.startswith("CounterNarrativeB"):
                   cnb += 1
               if current_component.startswith("CounterNarrativeC"):
                   cnc += 1
               if current_component.startswith("CounterNarrativeFree"):
                   cnf += 1
               if current_component.startswith(component):
                   component_text.append(delete_unwanted_chars(ann[2]))
                   if component.startswith("Premise"):
                       premise_codes.append(ann[0])
               if current_component.startswith("QuadrantType"):
                   info_splitted = current_component.split(" ")
                   if info_splitted[1].strip() in premise_codes:
                       type_of_premise = info_splitted[2].strip()
                       if type_of_premise == "fact":
                           fact = True
                       elif type_of_premise == "value":
                           value = True
                       else:
                           policy = True


        if not is_argumentative:
            non_argumentatives += 1
        elif len(component_text) > 0:
            have_component += 1
            component_words += len([word for component in component_text for  word in component.split()])

        if fact:
            type_fact += 1
        elif value:
            type_value += 1
        elif policy:
            type_policy += 1

    return [non_argumentatives, have_component, component_words, all_words, type_fact, type_value, type_policy, cna, cnb, cnc, cnf]

# TODO
#def labelsAgreementOverlap(annotator1, annotator2, percentage):
#    counting = False
#    for an1, an2 in zip(annotator1, annotator2):
#        if an1 == 1 or ann2 == 1:
#            if not counting:
#                counting = True



non_argumentative = 0
have = {}
words = {}
facts = {}
values = {}
policies = {}
all_words = 0
cna = 0
cnb = 0
cnc = 0
cnf = 0
for component in components:
    have[component] = 0
    words[component] = 0
    facts[component] = 0
    values[component] = 0
    policies[component] = 0
    non_argumentative = 0
    all_words = 0
    cna = 0
    cnb = 0
    cnc = 0
    cnf = 0
    print("Data for component " + component)
    for partition in filePatterns:
        results = labelComponentsFromAllExamples(partition, component)
        print(partition)
        print(results)
        non_argumentative += results[0]
        have[component] += results[1]
        words[component] += results[2]
        all_words += results[3]
        facts[component] += results[4]
        values[component] += results[5]
        policies[component] += results[6]
        cna += results[7]
        cnb += results[8]
        cnc += results[9]
        cnf += results[10]
    
print("Total words")
print(all_words)

print("Non argumentative")
print(non_argumentative)

print("Presence")
print(have)

print("words")
print(words)

print("facts")
print(facts)

print("values")
print(values)

print("policies")
print(policies)

print("CNA")
print(cna)

print("CNB")
print(cnb)

print("CNC")
print(cnc)

print("CNF")
print(cnf)
