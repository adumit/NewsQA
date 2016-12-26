import Dataset_Gen_Helper as helper

processed_train = helper.get_processed_train()
print("Done loading data...")

# This will generate a list of lists of sentences
all_sentences = [helper.clean_and_sentencize_entry(s_text, q) for s_text, q
                 in zip(processed_train['story_text'],
                        processed_train['question'])]
print("Data is in individual sentences...")

with open("sentence_per_line.txt", 'a') as f:
    for cleaned_entry in all_sentences:
        for s in cleaned_entry:
            f.write(s + "\n")
