from sklearn.metrics import hamming_loss, f1_score, accuracy_score

def report_scores(y, y_pred):
    score_str = ''
    hamm = hamming_loss(y, y_pred)
    score_str += '\n{:25s}{:>10.3f}\n'.format('Hamming Loss:', hamm)

    classes = [
        'volition',
        'prediction',
        'tact/rudeness',
        'necessity',
        'hypotheticality',
        'certainty',
        'agreement/disagreement',
        'contrariety',
        'source of knowledge',
        'uncertainty']

    score_str += '\nf1 scores \n -----------'
    f1 = f1_score(y, y_pred, average=None)
    scores = zip(classes, f1)
    for sc in sorted(scores, key=lambda s: s[1], reverse=True):
        score_str += '\n{:25s}{:>10.3f}'.format(sc[0].capitalize() + ':', sc[1])


    f1_macro = f1_score(y, y_pred, average='macro')
    f1_micro = f1_score(y, y_pred, average='micro')

    score_str += '\n\n{:25s}{:10.3f}'.format('Micro-f1 score:', f1_micro)
    score_str += ('\n{:25s}{:>10.3f}'.format('Macro-f1 score:', f1_macro))

    accuracy = accuracy_score(y, y_pred)
    score_str += '\n\n{:25s}{:10.3f}'.format('Accuracy', accuracy)

    return score_str
