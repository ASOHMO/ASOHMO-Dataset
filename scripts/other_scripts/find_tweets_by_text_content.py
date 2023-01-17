import glob

filePatterns = ["./data/HateEval/partition_{}/hate_tweet_*.txt".format(num) for num in range(1,11)]

patternsToMatch = ["./data/HateEval/partition_test_challenge_contrahate_for_evaluation/english/hate_tweet_*.txt"]

allFiles = []
for pattern in filePatterns:
    for f in glob.glob(pattern):
        allFiles.append(f)

allFilesToMatch = []
for pattern in patternsToMatch:
    for f in glob.glob(pattern):
        allFilesToMatch.append(f)

for f in allFilesToMatch:
    tweet = open(f, "r")
    annotations = open(f.replace("txt","ann"), "r")
    annotations_private = annotations.read()
    tweet_text = tweet.read()
    tweet_text_rpl = tweet_text.replace(" ","").replace("\n","")
    found = False
    for f2f in allFiles:
        new_tweet = open(f2f, "r")
        new_tweet_text = new_tweet.read()
        new_tweet_text_rpl = new_tweet_text.replace(" ","").replace("\n","")
        if tweet_text_rpl == new_tweet_text_rpl:
            found = True
            new_annotation = open(f2f.replace("txt","ann"),"r")
            new_annotation_txt = new_annotation.read()
            original_name = f2f.split("/")[-1]
            print("Changing {} for {}".format(f, original_name))
            new_location = open(original_name, "w")
            new_location_ann = open(original_name.replace("txt", "ann"), "w")
            private_ann = open(original_name.replace(".txt", "_private.ann"), "w")
            new_location.write(tweet_text)
            new_location_ann.write(new_annotation_txt)
            private_ann.write(annotations_private)
            new_location.close()
            new_location_ann.close()
            private_ann.close()
            break
    assert(found)

# for f in allFiles:
#     ann = open(f.replace("txt", "ann"), 'r')
#     is_arg = True
#     has_cn = False
#     for idx, line in enumerate(ann):
#         if len(line.split("\t")) <= 1:
#             print(line.split("\t"))
#             print(ann)
#         to_check = line.split("\t")[1]
#         if to_check.startswith("NonArgumentative"):
#             is_arg = False
#             break
#         if to_check.startswith("CounterNarrative"):
#             has_cn = True
#             break
#     if is_arg and not has_cn:
#         print(f)
#         print("----------------------")


