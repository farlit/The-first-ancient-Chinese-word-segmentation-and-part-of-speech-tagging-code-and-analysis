from sklearn.metrics import precision_score,recall_score,f1_score
def end_of_chunk(prev_tag, tag):
    """Checks if a chunk ended between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
    Returns:
        chunk_end: boolean.
    """
    chunk_end = False
    if prev_tag == 'S':
        chunk_end = True
    if prev_tag == 'E':
        chunk_end = True
    # pred_label中可能出现这种情形
    if prev_tag == 'B' and tag == 'B':
        chunk_end = True
    if prev_tag == 'B' and tag == 'S':
        chunk_end = True
    if prev_tag == 'B' and tag == 'O':
        chunk_end = True
    if prev_tag == 'I' and tag == 'B':
        chunk_end = True
    if prev_tag == 'I' and tag == 'S':
        chunk_end = True
    if prev_tag == 'I' and tag == 'O':
        chunk_end = True
    return chunk_end

def start_of_chunk(prev_tag, tag):
    """Checks if a chunk started between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
    Returns:
        chunk_start: boolean.
    """
    chunk_start = False
    if tag == 'B':
        chunk_start = True
    if tag == 'S':
        chunk_start = True
    if prev_tag == 'O' and tag == 'I':
        chunk_start = True
    if prev_tag == 'O' and tag == 'E':
        chunk_start = True
    if prev_tag == 'S' and tag == 'I':
        chunk_start = True
    if prev_tag == 'S' and tag == 'E':
        chunk_start = True
    if prev_tag == 'E' and tag == 'I':
        chunk_start = True
    if prev_tag == 'E' and tag == 'E':
        chunk_start = True
    return chunk_start

def get_entities(seq):
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]
    prev_tag = 'O'
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        tag = chunk[0]
        if end_of_chunk(prev_tag, tag):             #判断当前字符(tag)的前一个字符(prev_tag)是不是结束字符
            chunks.append((begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag):           #判断当前字符是不是开始字符
            begin_offset = i
        prev_tag = tag
    return chunks

def f1_score_seg(y_true, y_pred):
    true_entities = set(get_entities(y_true))
    pred_entities = set(get_entities(y_pred))
    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)
    # print(nb_correct)  # 99569
    # print(nb_pred)  # 236817
    # print(nb_true)  # 104519

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0
    return score, p, r

def f1_score_pos(y_true, y_pred):

    y_true = [j for i in y_true for j in i]
    y_pred = [j for i in y_pred for j in i]
    score = f1_score(y_true, y_pred, average='micro')
    p = precision_score(y_true, y_pred, average='micro')
    r = recall_score(y_true, y_pred, average='micro')
    return score, p, r