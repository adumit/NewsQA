import pandas as pd
import Dataset_Gen_Helper as helper

processed_train = helper.get_processed_train()
print("Done loading data...")

# This will generate a list of lists of (sentence, question) pairs
sq_pairs = [helper.pull_out_qt_data(s_text, ans, q)
             for s_text, ans, q
             in zip(processed_train['story_text'],
                    processed_train['answer_char_ranges'],
                    processed_train['question'])]

all_sq_pairs = []
for sq_pair_list in sq_pairs:
    all_sq_pairs += sq_pair_list

print(len(all_sq_pairs))

sentences = [sq[0] for sq in all_sq_pairs if sq is not None]
questions = [sq[1] for sq in all_sq_pairs if sq is not None]

with open("./Text_Data/story_text_per_line_lt65", 'ab') as fs:
    with open("./Text_Data/question_per_line_lt65", 'ab') as fq:
        for i in range(len(sentences)):
            if len(sentences[i].split()) < 65 and len(questions[i].split()) < 15:
                fs.write((sentences[i] + "\n").encode('utf-8'))
                fq.write((questions[i] + "\n").encode('utf-8'))


# sq_df = pd.DataFrame(data={"sentences": sentences, "questions":questions})
# sq_df.to_csv("Sentence_Question_train.csv", index=False)
