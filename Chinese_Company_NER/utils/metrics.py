import os
import config
import logging


def get_entities(seq):
    """
    Gets entities from sequence.

    Args:
        seq (list): sequence of labels.

    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).

    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ["O"]]
    prev_tag = "O"
    prev_type = ""
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ["O"]):
        tag = chunk[0]
        type_ = chunk.split("-")[-1]

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == "S":
        chunk_end = True
    # pred_label中可能出现这种情形
    if prev_tag == "B" and tag == "B":
        chunk_end = True
    if prev_tag == "B" and tag == "S":
        chunk_end = True
    if prev_tag == "B" and tag == "O":
        chunk_end = True
    if prev_tag == "I" and tag == "B":
        chunk_end = True
    if prev_tag == "I" and tag == "S":
        chunk_end = True
    if prev_tag == "I" and tag == "O":
        chunk_end = True

    if prev_tag != "O" and prev_tag != "." and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == "B":
        chunk_start = True
    if tag == "S":
        chunk_start = True

    if prev_tag == "S" and tag == "I":
        chunk_start = True
    if prev_tag == "O" and tag == "I":
        chunk_start = True

    if tag != "O" and tag != "." and prev_type != type_:
        chunk_start = True

    return chunk_start


def f1_score(y_true, y_pred, mode="dev"):
    """Compute the F1 score.

    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::

        F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.

    Returns:
        score : float.

    Example:
        y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        f1_score(y_true, y_pred)
        0.50
    """
    true_entities = set(get_entities(y_true))
    pred_entities = set(get_entities(y_pred))
    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0
    if mode == "dev":
        return score
    else:
        f_score = {}
        for label in config.labels:
            true_entities_label = set()
            pred_entities_label = set()
            for t in true_entities:
                if t[0] == label:
                    true_entities_label.add(t)
            for p in pred_entities:
                if p[0] == label:
                    pred_entities_label.add(p)
            nb_correct_label = len(true_entities_label & pred_entities_label)
            nb_pred_label = len(pred_entities_label)
            nb_true_label = len(true_entities_label)

            p_label = nb_correct_label / nb_pred_label if nb_pred_label > 0 else 0
            r_label = nb_correct_label / nb_true_label if nb_true_label > 0 else 0
            score_label = (
                2 * p_label * r_label / (p_label + r_label)
                if p_label + r_label > 0
                else 0
            )
            f_score[label] = score_label
        return f_score, score


def bad_case(y_true, y_pred, data):
    if not os.path.exists(config.case_dir):
        os.system(r"touch {}".format(config.case_dir))  # 调用系统命令行来创建文件
    output = open(config.case_dir, "w")
    for idx, (t, p) in enumerate(zip(y_true, y_pred)):
        if t == p:
            continue
        else:
            output.write("bad case " + str(idx) + ": \n")
            output.write("sentence: " + str(data[idx]) + "\n")
            output.write("golden label: " + str(t) + "\n")
            output.write("model pred: " + str(p) + "\n")
    logging.info("--------Bad Cases reserved !--------")


def predict_output(y_pred, data):
    # if not os.path.exists(config.predict_seq_output_dir):
    # os.system(r"touch {}".format(config.predict_seq_output_dir))
    # if not os.path.exists(config.predict_dict_output_dir):
    # os.system(r"touch {}".format(config.predict_dict_output_dir))
    # if not os.path.exists(config.predict_dict_output_dir):
    # os.system(r"touch {}".format(config.predict_entity_list_dir))
    # seq_output = open(config.predict_seq_output_dir, 'w')
    # dict_output = open(config.predict_dict_output_dir, 'w')
    # entity_list_output = open(config.predict_entity_list_dir, 'w')

    all_end = []
    for idx, p in enumerate(y_pred):

        print(data[idx], p)
        try:
            output = convert_to_dict(data[idx], p)
        except:
            all_end.append([[], []])
            continue
        end = [output["text"]]
        # seq_output.write("output " + str(idx) + ": \n")
        # seq_output.write("sentence: " + str(data[idx]) + "\n")
        # seq_output.write("model pred: " + str(p) + "\n")
        # dict_output.write(str(output) + "\n")
        words_ = []
        for word in output["label"]["element"]:
            # entity_list_output.write(str(word) + '\n')
            words_.append(str(word))
        end.append(words_)
        all_end.append(end)
    logging.info("--------Predict finished !--------")
    return all_end


def convert_to_dict(sentence, pred):
    word_idx_list = list()
    output_dict = dict()
    sentence = "".join(sentence)
    sentence = sentence.replace("[UNK]", "X")
    output_dict["text"] = sentence
    output_dict["label"] = dict()
    output_dict["label"]["element"] = dict()
    tag = 0
    for i in range(len(pred)):
        if pred[i][0] == "B" or pred[i][0] == "I":
            if tag == 0:
                word_idx_list.append(i)
            elif i == len(pred) - 1:
                word_idx_list.append(i)
            tag = 1
        if pred[i][0] == "O":
            if tag == 1:
                word_idx_list.append(i - 1)
            tag = 0
    assert len(word_idx_list) % 2 == 0
    entity_num = len(word_idx_list) // 2
    for i in range(entity_num):
        try:
            output_dict["label"]["element"][
                sentence[word_idx_list[2 * i] : word_idx_list[2 * i + 1] + 1]
            ].append([word_idx_list[2 * i], word_idx_list[2 * i + 1]])
        except:
            output_dict["label"]["element"][
                sentence[word_idx_list[2 * i] : word_idx_list[2 * i + 1] + 1]
            ] = []
            output_dict["label"]["element"][
                sentence[word_idx_list[2 * i] : word_idx_list[2 * i + 1] + 1]
            ].append([word_idx_list[2 * i], word_idx_list[2 * i + 1]])
    return output_dict


def predict_output_pre(y_pred, data):
    all_end = []
    for idx, p in enumerate(y_pred):
        sentence = data[idx]
        for i in range(len(sentence)):
            if sentence[i] == "[UNK]":
                sentence[i] = ""

        sentence_str = "".join(sentence)
        if len(data[idx]) != len(p):
            all_end.append([sentence_str, {}])
            continue

        try:
            output = convert_to_dict_pre(sentence, p)
            all_end.append([sentence_str, output])
        except:
            all_end.append([sentence_str, {}])

    logging.info("--------Predict finished !--------")
    return all_end


def convert_to_dict_pre(sentence, pred):
    # ['g', 'u', 'r', 'm', 'a', 'n', '：', '元', '宇', '宙', '将', '会', '是', '苹', '果', 'a', 'r', '/', 'v', 'r', '头', '显', '的', '"', '禁', '区', '"']
    # ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-product', 'I-product', 'I-product', 'I-product', 'I-product', 'I-product', 'I-product', 'I-product', 'O', 'O', 'O', 'O', 'O', 'O']
    # print(sentence, pred)
    i = 0
    dic = {}
    while i < len(pred):
        # print(pred[i])
        if pred[i][0] == "B":
            type = pred[i][2:]  # product
            if type not in dic:
                dic[type] = []
            s = sentence[i]
            start = i
            i += 1
            while i < len(pred) and pred[i] == "I-" + type:
                s += sentence[i]
                i += 1
            end = i
            i -= 1
            dic[type].append([s, start, end])
        elif pred[i][0] == "S":
            type = pred[i][2:]  # product
            if type not in dic:
                dic[type] = []
            dic[type].append([sentence[i], i, i + 1])

        i += 1
    return dic


# sentence = ["特", "斯", "拉", "搞", "死", "了", "苹", "果"]
# pred = ["B-company", "I-company", "I-company", "O", "O", "O", "B-company", "I-company"]
# print(convert_to_dict_pre(sentence, pred))
