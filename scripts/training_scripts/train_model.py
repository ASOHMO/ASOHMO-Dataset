import glob
from transformers import AutoTokenizer
from datasets import Dataset
from transformers import DataCollatorForTokenClassification, DataCollatorWithPadding
from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
import torch
from transformers import EvalPrediction
from sklearn import metrics
import argparse
from transformers import EarlyStoppingCallback
import random


parser = argparse.ArgumentParser(description="Train models for identifying argumentative components inside the ASFOCONG dataset")
parser.add_argument('components', type=str, nargs='+', help="Name of the component that wants to be identified")
parser.add_argument('--modelname', type=str, default="roberta-base", help="Name of the language model to be downloaded from huggingface")
parser.add_argument('--lr', type=float, default=2e-05, help="Learning rate for training the model. Default value is 2e-05")
parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training and evaluation. Default size is 16")
parser.add_argument('--add_annotator_info', type=bool, default=False, help="For Pivot and Collective add information about premises and Property respectively that an annotator would have when annotating these components")
parser.add_argument('--type_of_premise', type=bool, default=False, help="If true, model will be trained to predict the type of premises. If true, only valid components are Justification and Conclusion")
parser.add_argument('--simultaneous_components', type=bool, default=False, help="Set to true if trying to do joint predictions")
parser.add_argument('--multilingual', type=bool, default=False, help="Set to true if using both english and spanish datasets. Both Training and Test datasets will have both languages")
parser.add_argument('--joint_premises', type=int, default=0, help="If true, this script will predict type of premise disregarding if the premise is Justification of Conclusion")
parser.add_argument('--crosslingual', type=bool, default=False, help="Set to true if using both english and spanish datasets. English dataset will be used for train and dev and Spanish will be used for testing")
parser.add_argument('--only_if_present', type=bool, default=False, help="Only train and test with examples that have the component they are trying to predict. Changes the construction of the datasets for Collective, Property and pivot")
parser.add_argument('--predict_if_present', type=bool, default=False, help="For each tweet, predict if component is present. Only works for Collective/Property or Pivot")

args = parser.parse_args()


LEARNING_RATE = args.lr
NUMBER_OF_PARTITIONS = 10
device = torch.device("cpu")
BATCH_SIZE = args.batch_size
EPOCHS = 20 * (BATCH_SIZE / 16)
MODEL_NAME = args.modelname
REP=0
FOLDS=3
SEQ_LENGTH = 127
components = args.components
component = components[0]
add_annotator_info = args.add_annotator_info
type_of_premise = args.type_of_premise or args.joint_premises
simultaneous_components = args.simultaneous_components
crosslingual = args.crosslingual
multilingual = args.multilingual or crosslingual
joint_premises = args.joint_premises
quadrant_types_to_label = {"fact": 0, "value": 1, "policy": 2}
only_if_present = args.only_if_present
predict_if_present = args.predict_if_present

def compute_metrics_f1(p: EvalPrediction):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids

    if not type_of_premise and component != "Argumentative" and not predict_if_present:
        true_labels = [[str(l) for l in label if l != -100] for label in labels]
        true_predictions = [
            [str(p) for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(preds, labels)
        ]
        all_true_labels = [l for label in true_labels for l in label]
        all_true_preds = [p for preed in true_predictions for p in preed]
        if simultaneous_components:
            avrge = "macro"
            f1_all = metrics.f1_score(all_true_labels, all_true_preds, average=None, labels=['0', '1', '2'])
            precision_all = metrics.precision_score(all_true_labels, all_true_preds, average=None, labels=['0', '1', '2'])
            recall_all = metrics.recall_score(all_true_labels, all_true_preds, average=None, labels=['0', '1', '2'])
        else:
            avrge = "binary"
    else:
        all_true_labels = [str(label) for label in labels]
        all_true_preds = [str(pred) for pred in preds]
        if type_of_premise:
            avrge = "macro"
            f1_all = metrics.f1_score(all_true_labels, all_true_preds, average=None, labels=['0','1','2'])
        else:
            avrge = "binary"


    f1 = metrics.f1_score(all_true_labels, all_true_preds, average=avrge, pos_label='1')

    acc = metrics.accuracy_score(all_true_labels, all_true_preds)

    recall = metrics.recall_score(all_true_labels, all_true_preds, average=avrge, pos_label='1')

    precision = metrics.precision_score(all_true_labels, all_true_preds, average=avrge, pos_label='1')

    f1_micro = metrics.f1_score(all_true_labels, all_true_preds, average="micro")

    recall_micro = metrics.recall_score(all_true_labels, all_true_preds, average="micro")

    precision_micro = metrics.precision_score(all_true_labels, all_true_preds, average="micro")

    confusion_matrix = metrics.confusion_matrix(all_true_labels, all_true_preds)


    w = open("./results_{}_{}_{}_{}_{}-metrics".format(LEARNING_RATE, MODEL_NAME.replace("/", "-"), BATCH_SIZE, component, REP), "a")

    w.write("{},{},{},{},{},{},{}\n".format(str(acc), str(f1), str(precision), str(recall), str(f1_micro), str(precision_micro), str(recall_micro)))
    w.close()

    ans = {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': str(confusion_matrix),
    }

    if type_of_premise or simultaneous_components:
        ans['f1_all'] = str(f1_all)
        if simultaneous_components:
            ans['precision_all'] = str(precision_all)
            ans['recall_all'] = str(recall_all)

    return ans



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

def delete_unwanted_chars(text):
    return text.replace("\n", "").replace("\t", "").replace(".", "").replace(",", "").replace("!", "").replace("#", "").replace('“', '"').replace('”', '"').replace('…', '').replace("’", "").replace("–", " ").replace("‘", "").replace("—", "").replace("·", "")

def getLabel(label):
    if label == "O":
        return 0
    else:
        return 1


def labelComponentsFromAllExamples(filePatterns, componentt, multidataset = False, add_annotator_info = False, isTypeOfPremise = False, multiple_components = False, joint_premises=False, only_if_present=False, predict_if_present=False):
    all_tweets = []
    all_labels = []
    if multidataset:
        datasets = []
    for filePattern in filePatterns:
        for f in glob.glob(filePattern):
            conll_file = open(f, 'r')
            tweet = []
            if not isTypeOfPremise:
                labels = []
            else:
                labels = -1
            if add_annotator_info:
                if componentt == "Collective":
                    property_text = []
                if component == "Property":
                    collective_text = []
                if componentt == "pivot":
                    justification_text = []
                    conclusion_text = []
            if only_if_present or predict_if_present:
                text_is_present = False
            is_argumentative = True
            for idx, line in enumerate(conll_file):
                line_splitted = line.split("\t")
                word = line_splitted[0]
                if not isTypeOfPremise:
                    tweet.append(word)
                else:
                    if componentt == "Premise2Justification":
                        if line_splitted[2] != "O":
                            tweet.append(word)
                    elif componentt == "Premise1Conclusion":
                        if line_splitted[3] != "O":
                            tweet.append(word)
                if line_splitted[1] != "O" or not is_argumentative:
                    is_argumentative = False
                    continue


                if isTypeOfPremise:
                    if componentt == "Premise2Justification":
                        if line_splitted[2] != "O":
                            labels = quadrant_types_to_label[line_splitted[7].replace("\n", "")]
                    elif componentt == "Premise1Conclusion":
                        if line_splitted[3] != "O":
                            labels = quadrant_types_to_label[line_splitted[8].replace("\n", "")]
                elif multiple_components:
                    # getLabel returns 0 or 1 depending on if the current word belongs to the correspondent component.
                    # If were the unlikley case of a word belonging to both Collective and Property,
                    # the max function will return the label for Property. If it happened to belong to both Justification
                    # and Conclusion, it will be labeled as Conclusion.
                    if componentt == "Collective-Property":
                        col = getLabel(line_splitted[4])
                        prop = getLabel(line_splitted[5]) * 2
                        labels.append(max(col, prop))
                    if componentt == "Premises":
                        just = getLabel(line_splitted[2])
                        conc = getLabel(line_splitted[3]) * 2
                        labels.append(max(just, conc))
                else:
                    if componentt == "Premise2Justification":
                        labels.append(getLabel(line_splitted[2]))
                    elif componentt == "Premise1Conclusion":
                        labels.append(getLabel(line_splitted[3]))
                    else:
                        if componentt == "Collective":
                            lbll = getLabel(line_splitted[4])
                            labels.append(lbll)
                            if add_annotator_info and getLabel(line_splitted[5]) == 1:
                                property_text.append(word)
                        elif componentt == "Property":
                            lbll = getLabel(line_splitted[5])
                            labels.append(lbll)
                            if add_annotator_info and getLabel(line_splitted[4]) == 1:
                                collective_text.append(word)
                        elif componentt == "pivot":
                            lbll = getLabel(line_splitted[6])
                            labels.append(lbll)
                            if add_annotator_info and getLabel(line_splitted[2]) == 1:
                                justification_text.append(word)
                            if add_annotator_info and getLabel(line_splitted[3]) == 1:
                                conclusion_text.append(word)
                        
                        if (only_if_present or predict_if_present) and lbll == 1:
                            text_is_present = True

            if componentt == "Argumentative":
                labels = 1 if is_argumentative else 0
            if not is_argumentative and componentt != "Argumentative":
                continue
            if predict_if_present:
                labels = 1 if text_is_present else 0
            if only_if_present and not text_is_present:
                continue
            if isTypeOfPremise:
                assert(labels >= 0)
            if add_annotator_info:
                to_add = []
                if componentt == "Collective":
                    to_add = ["[SEP]", "Property:"] + property_text
                if component == "Property":
                    to_add = ["[SEP]", "Collective:"] + collective_text
                if componentt == "pivot":
                    to_add = ["[SEP]", "Justification:"] + justification_text + ["[SEP]", "Conclusion:"] + conclusion_text
                tweet += to_add
                labels += [-100] * len(to_add)

            if multidataset:
                dicc = {"tokens": [tweet], "labels": [labels]}
                datasets.append([Dataset.from_dict(dicc), tweet])
            else:
                all_tweets.append(tweet)
                all_labels.append(labels)


    if multidataset:
        return datasets

    if joint_premises > 0:
        return all_tweets, all_labels    

    ans = {"tokens": all_tweets, "labels": all_labels}
    return Dataset.from_dict(ans)


def tokenize_and_align_labels(dataset, tokenizer, is_multi = False, is_bertweet=False, one_label_per_example=False):
    def tokenize_and_align_labels_one_label(example):
        return tokenizer(example["tokens"], truncation=True, is_split_into_words=True)

    def tokenize_and_align_labels_per_example(example):
        tokenized_inputs = tokenizer(example["tokens"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(example[f"labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs


    def tokenize_and_align_labels_per_example_bertweet(example):
        tkns = example["tokens"]
        labels = example["labels"]
        if len(tkns) == 0 and len(labels) == 0:
            return {"input_ids": [], "labels": [], "attention_mask": []}
        tokenized_input = tokenizer(tkns, truncation=True, is_split_into_words=True)
        label_ids = [-100]
        accum = []
        for word, label in zip(tkns, labels):
            if word != "":
                tokens = tokenizer(word).input_ids
                label_ids.append(label)
                for i in range(len(tokens)-3):
                    label_ids.append(-100)
                accum.append(word)
                tmp_accum = tokenizer(accum, is_split_into_words=True)
                assert(len(tmp_accum.input_ids) == len(label_ids) + 1)
        if len(label_ids) > SEQ_LENGTH:
            label_ids = label_ids[:SEQ_LENGTH]
        label_ids.append(-100)
        assert(len(tokenized_input.input_ids) == len(label_ids))
        assert(len(tokenized_input.input_ids) == len(tokenized_input.attention_mask))
        return {"input_ids": tokenized_input.input_ids, "labels": label_ids, "attention_mask": tokenized_input.attention_mask}

    if one_label_per_example:
        function_to_apply = tokenize_and_align_labels_one_label
    else:
        function_to_apply = tokenize_and_align_labels_per_example
        if is_bertweet:
            function_to_apply = tokenize_and_align_labels_per_example_bertweet
            if is_multi:
                return [{"dataset": data[0].map(function_to_apply), "text": data[1]} for data in dataset]
            return dataset.map(function_to_apply)
    if is_multi:
        return [{"dataset": data[0].map(function_to_apply, batched=True), "text": data[1]} for data in dataset]
    return dataset.map(function_to_apply, batched=True)



def train(model, tokenizer, train_partition_patterns, dev_partition_patterns, test_partition_patterns, component, is_bertweet=False, add_annotator_info=False, is_type_of_premise=False, multiple_components = False, joint_premises = False, only_if_present = False, predict_if_present = False):

    if joint_premises > 0:
        just_tweets, just_labels = labelComponentsFromAllExamples(train_partition_patterns, "Premise2Justification", add_annotator_info=add_annotator_info, isTypeOfPremise=joint_premises, multiple_components=multiple_components, joint_premises=joint_premises, only_if_present=only_if_present, predict_if_present=predict_if_present)
        conc_tweets, conc_labels = labelComponentsFromAllExamples(train_partition_patterns, "Premise1Conclusion", add_annotator_info=add_annotator_info, isTypeOfPremise=joint_premises, multiple_components=multiple_components, joint_premises=joint_premises, only_if_present=only_if_present, predict_if_present=predict_if_present)
        twts = just_tweets + conc_tweets
        lbls = just_labels + conc_labels
        dtst_dict = {"tokens": twts, "labels": lbls}
        training_set = tokenize_and_align_labels(Dataset.from_dict(dtst_dict), tokenizer, is_bertweet = is_bertweet, one_label_per_example=(is_type_of_premise or component == "Argumentative" or predict_if_present))

        just_tweets_dev, just_labels_dev = labelComponentsFromAllExamples(dev_partition_patterns, "Premise2Justification", add_annotator_info=add_annotator_info, isTypeOfPremise=joint_premises, multiple_components=multiple_components, joint_premises=joint_premises, only_if_present=only_if_present, predict_if_present=predict_if_present)
        conc_tweets_dev, conc_labels_dev = labelComponentsFromAllExamples(dev_partition_patterns, "Premise1Conclusion", add_annotator_info=add_annotator_info, isTypeOfPremise=joint_premises, multiple_components=multiple_components, joint_premises=joint_premises, only_if_present=only_if_present, predict_if_present=predict_if_present)
        twts_dev = just_tweets_dev + conc_tweets_dev
        lbls_dev = just_labels_dev + conc_labels_dev
        dtst_dict_dev = {"tokens": twts_dev, "labels": lbls_dev}
        dev_set = tokenize_and_align_labels(Dataset.from_dict(dtst_dict_dev), tokenizer, is_bertweet = is_bertweet, one_label_per_example=(is_type_of_premise or component == "Argumentative" or predict_if_present))

        just_tweets_test, just_labels_test = labelComponentsFromAllExamples(test_partition_patterns, "Premise2Justification", add_annotator_info=add_annotator_info, isTypeOfPremise=joint_premises, multiple_components=multiple_components, joint_premises=joint_premises, only_if_present=only_if_present, predict_if_present=predict_if_present)
        conc_tweets_test, conc_labels_test = labelComponentsFromAllExamples(test_partition_patterns, "Premise1Conclusion", add_annotator_info=add_annotator_info, isTypeOfPremise=joint_premises, multiple_components=multiple_components, joint_premises=joint_premises, only_if_present=only_if_present, predict_if_present=predict_if_present)
        if joint_premises == 1:
            twts_test = just_tweets_test + conc_tweets_test
            lbls_test = just_labels_test + conc_labels_test
        elif joint_premises == 2:
            twts_test = just_tweets_test
            lbls_test = just_labels_test
        elif joint_premises == 3:
            twts_test = conc_tweets_test
            lbls_test = conc_labels_test
        dtst_dict_test = {"tokens": twts_test, "labels": lbls_test}
        test_set = tokenize_and_align_labels(Dataset.from_dict(dtst_dict_test), tokenizer, is_bertweet = is_bertweet, one_label_per_example=(is_type_of_premise or component == "Argumentative" or predict_if_present))
        test_set_one_example = tokenize_and_align_labels(labelComponentsFromAllExamples(test_partition_patterns, component, multidataset = True, add_annotator_info=add_annotator_info, isTypeOfPremise=is_type_of_premise, multiple_components=multiple_components), tokenizer, is_multi = True, is_bertweet = is_bertweet, one_label_per_example=(is_type_of_premise or component == "Argumentative"))

    else:
        training_set = tokenize_and_align_labels(labelComponentsFromAllExamples(train_partition_patterns, component, add_annotator_info=add_annotator_info, isTypeOfPremise=is_type_of_premise, multiple_components=multiple_components, only_if_present=only_if_present, predict_if_present=predict_if_present), tokenizer, is_bertweet = is_bertweet, one_label_per_example=(is_type_of_premise or component == "Argumentative" or predict_if_present))
        dev_set = tokenize_and_align_labels(labelComponentsFromAllExamples(dev_partition_patterns, component, add_annotator_info=add_annotator_info, isTypeOfPremise=is_type_of_premise, multiple_components=multiple_components, only_if_present=only_if_present, predict_if_present=predict_if_present), tokenizer, is_bertweet = is_bertweet, one_label_per_example=(is_type_of_premise or component == "Argumentative" or predict_if_present))
        test_set = tokenize_and_align_labels(labelComponentsFromAllExamples(test_partition_patterns, component, add_annotator_info=add_annotator_info, isTypeOfPremise=is_type_of_premise, multiple_components=multiple_components, only_if_present=only_if_present, predict_if_present=predict_if_present), tokenizer, is_bertweet = is_bertweet, one_label_per_example=(is_type_of_premise or component == "Argumentative" or predict_if_present))
        test_set_one_example = tokenize_and_align_labels(labelComponentsFromAllExamples(test_partition_patterns, component, multidataset = True, add_annotator_info=add_annotator_info, isTypeOfPremise=is_type_of_premise, multiple_components=multiple_components), tokenizer, is_multi = True, is_bertweet = is_bertweet, one_label_per_example=(is_type_of_premise or component == "Argumentative"))
    
    training_args = TrainingArguments(
        output_dir="./results_eval_{}_{}".format(MODEL_NAME.replace("/", "-"), component),
        evaluation_strategy="steps",
        eval_steps=10,
        save_total_limit=8,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.05,
        report_to="none",
        metric_for_best_model='f1',
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_set,
        eval_dataset=dev_set,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics= compute_metrics_f1,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=10)]
    ) 

    trainer.train()

    model_settings = MODEL_NAME.replace("/", "-")
    if crosslingual:
        model_settings += "-xl"
    elif multilingual:
        model_settings += '-mix'
    results = trainer.predict(test_set)
    if not type_of_premise:
        if add_annotator_info:
            suffix = "_ADDED-INFO"
        elif only_if_present:
            suffix = "_ONLY-PRESENTS"
        elif predict_if_present:
            suffix = "_PREDICT-IF_PRESENT"
        else:
            suffix = ""
        filename = "./results_test_{}_{}_{}_{}_{}{}".format(LEARNING_RATE, model_settings, BATCH_SIZE, REP, component, suffix)
    else:
        if joint_premises == 1:
            suffix = "joint-premises"
        elif joint_premises == 2:
            suffix = "tested-with-just"
        elif joint_premises == 3:
            suffix = "tested-with-conc"
        else:
            suffix = "type-of-premise"
        filename = "./results_test_{}_{}_{}_{}_{}_{}".format(LEARNING_RATE, model_settings, BATCH_SIZE, REP, component, suffix)
    with open(filename, "w") as writer:
        if type_of_premise:
            writer.write("{},{},{},{},{}\n".format(results.metrics["test_accuracy"], results.metrics["test_f1"], results.metrics["test_precision"], results.metrics["test_recall"], results.metrics["test_f1_all"]))
        elif multiple_components:
            writer.write("{},{},{},{},{},{},{}\n".format(results.metrics["test_accuracy"], results.metrics["test_f1"], results.metrics["test_precision"], results.metrics["test_recall"], results.metrics["test_f1_all"], results.metrics["test_precision_all"], results.metrics["test_recall_all"]))
        else:
            writer.write("{},{},{},{}\n".format(results.metrics["test_accuracy"], results.metrics["test_f1"], results.metrics["test_precision"], results.metrics["test_recall"]))
        writer.write("{}".format(str(results.metrics["test_confusion_matrix"])))

    examples_filename = "./examples_test_{}_{}_{}_{}_{}_{}".format(LEARNING_RATE, MODEL_NAME.replace("/", "-"), BATCH_SIZE, REP, component, suffix)
    with open(examples_filename, "w") as writer:
        for dtset in test_set_one_example:
            result = trainer.predict(dtset["dataset"])
            preds = result.predictions.argmax(-1)
            labels = result.label_ids

            if not type_of_premise and component != "Argumentative" and not predict_if_present:
                true_labels = [[str(l) for l in label if l != -100] for label in labels]
                true_predictions = [
                    [str(p) for (p, l) in zip(prediction, label) if l != -100]
                    for prediction, label in zip(preds, labels)
                ]
                all_true_labels = [l for label in true_labels for l in label]
                all_true_preds = [p for preed in true_predictions for p in preed]
            else:
                all_true_labels = [str(label) for label in labels]
                all_true_preds = [str(pred) for pred in preds]
            if not len(all_true_preds) == len(all_true_labels):
                print(all_true_preds)
                print(all_true_labels)
                print(dtset["dataset"]["tokens"][0])
                assert (len(all_true_preds) == len(all_true_labels))
            comparison = [(truth, pred) for truth, pred in zip(all_true_labels, all_true_preds) if truth != -100]
            writer.write("Tweet:\n")
            writer.write("{}\n".format(dtset["dataset"]["tokens"][0]))
            if component == "Argumentative":
                writer.write("{} - {}".format(comparison[0][0], comparison[0][1]))
            else:
                for word, pair in zip(dtset["dataset"]["tokens"][0], comparison):
                    writer.write("{}\t\t\t{}\t{}\n".format(word, pair[0], pair[1]))
            writer.write("-------------------------------------------------------------------------------\n")


if multilingual:
    filePatterns = ["./datasets_CoNLL/english/hate_tweet_*.conll", "./datasets_CoNLL/spanish/hate_tweet_*.conll"]
else:
    filePatterns = ["./datasets_CoNLL/english/hate_tweet_*.conll"]

dataset_combinations = []
if crosslingual:
    trainFiles = []
    testFiles = []
    for f in glob.glob(filePatterns[0]):
        trainFiles.append(f)
    for f in glob.glob(filePatterns[1]):
        testFiles.append(f)
    
    for i in range(FOLDS):
        trainFilesCp = trainFiles.copy()
        testFilesCp = testFiles.copy()
        random.Random(41 + i).shuffle(trainFilesCp)
        dataset_combinations.append([trainFilesCp[:850], trainFilesCp[850:], testFilesCp])

else:
    allFiles = []
    for pattern in filePatterns:
        for f in glob.glob(pattern):
            allFiles.append(f)

    for i in range(FOLDS):
        allFilesCp = allFiles.copy()
        random.Random(41 + i).shuffle(allFilesCp)
        if multilingual:
            dataset_combinations.append([allFilesCp[:890], allFilesCp[890:1016], allFilesCp[1016:]])
        else:
            dataset_combinations.append([allFilesCp[:770], allFilesCp[770:870], allFilesCp[870:]])

for combination in dataset_combinations:
    REP = REP + 1
    for cmpnent in components:
        component = cmpnent
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
        if not type_of_premise and component != "Argumentative" and not predict_if_present:
            data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
            if not simultaneous_components:
                output_num = 2
            else:
                output_num = 3
            model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=output_num)
            
        else:
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
            if type_of_premise:
                 output_num = 3
            else:
                 output_num = 2
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=output_num)

        model.to(device)
        train(model, tokenizer, combination[0], combination[1], combination[2], cmpnent, is_bertweet = MODEL_NAME == "vinai/bertweet-base", add_annotator_info=add_annotator_info, is_type_of_premise = type_of_premise, multiple_components=simultaneous_components, joint_premises=joint_premises, only_if_present=only_if_present, predict_if_present=predict_if_present)


