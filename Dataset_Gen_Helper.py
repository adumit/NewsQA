import pandas as pd
import numpy as np
import re


def clean_text(raw_text):
    """
    Clean the text by removing the CNN header piece and then removing all
     non-characters and white space (basically punctuation) and making
     everything lowercase
    """
    processed_text = re.sub("(.+)?(\(CNN\) +-- )", '', raw_text)
    processed_text = re.sub(r'[^\w\s]','', processed_text.lower())
    return processed_text


def split_sentences(story_text):
    """
    Put in artificial splitter in place of each \r or \n character and then
     split based on those. List comprehension over the list to remove all empty
     strings that result from multiple splittings.
    """
    sentences = [x for x in
                 re.split('_SPLITTER_',
                          re.sub('[\r\n]', "_SPLITTER_", story_text))
                 if x != '']
    return sentences


def find_matching_sentence(indexed_sentences, answer_index, question):
    """
    Find the sentence from the story that contains the answer and return a
     tuple of (clean sentence, clean question)
    """
    int_indices = [int(x) for x in answer_index.split(":")]
    for k in indexed_sentences.keys():
        # Currently, only look for answers that are contained in a single sentence
        if int_indices[0] in k and int_indices[1] in k:
            return (clean_text(indexed_sentences[k]), clean_text(question))


def pull_out_qt_data(story_text, answer_index_string, question):
    """
    Takes in the story text, string of answers from humans, and the question.
    Returns a list where each entry is the sentence from the story that the answer
    came from. All data in the return is cleaned
    """
    sentences = split_sentences(story_text)
    # Split for each person
    answers_per_person = answer_index_string.split("|")
    sentence_index_pairs = {}
    current_char_index = 0
    for sentence in sentences:
        sentence_index_pairs[range(current_char_index, current_char_index + len(
            sentence) - 1)] = sentence
        current_char_index += len(sentence)
    question_text_pairs = []
    for single_answer in answers_per_person:
        answer_indices = single_answer.split(",")
        [question_text_pairs.append(
            find_matching_sentence(sentence_index_pairs, ai, question)) for ai
         in answer_indices
         if ai != "None"]
    return question_text_pairs


def clean_and_sentencize_entry(story_text, question):
    """
    Used to turn a story text and question into a list of clean sentences.
    Returns a list with each entry being a sentence from the story or the question text
    """
    sentences = split_sentences(story_text)
    return [clean_text(s) for s in sentences] + [clean_text(question)]


def testnan(input_string):
    try:
        np.isnan(input_string)
        return True
    except:
        return False


def tryFloat(input_string):
    try:
        f = float(input_string)
        return f
    except:
        return -1.0


def get_processed_train():
    raw_train = pd.read_csv("newsqa/maluuba/newsqa/split_data/train.csv")
    answer_present_train = raw_train[raw_train['is_answer_absent'] < 0.5]
    print(answer_present_train.shape)

    answer_present_train['question_bad_float'] = [tryFloat(x) for x in answer_present_train['is_question_bad']]
    processed_train = answer_present_train[answer_present_train['question_bad_float'] < 0.5]

    print(processed_train.shape)
    return processed_train