from sklearn.metrics import hamming_loss, f1_score, accuracy_score
import numpy as np


def report_scores(y, y_pred):

    hamm = hamming_loss(y, y_pred)
    f1 = f1_score(y, y_pred, average=None)
    f1_macro = f1_score(y, y_pred, average='macro')
    f1_micro = f1_score(y, y_pred, average='micro')
    accuracy = accuracy_score(y, y_pred)

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

    scores = zip(classes, f1, np.add.reduce(y))
    score_str = '\n{:25s}{:>10.3f}\n'.format('Hamming Loss:', hamm)
    score_str += '\nf1 scores \n -----------'
    for label, score, freq in sorted(scores, key=lambda s: s[0], reverse=False):
        score_str += '\n{:25s}{:>10.3f}{:>10d}'.format(label.capitalize() + ':', score, freq)
    score_str += '\n\n{:25s}{:10.3f}'.format('Micro-f1 score:', f1_micro)
    score_str += ('\n{:25s}{:>10.3f}'.format('Macro-f1 score:', f1_macro))


    score_str += '\n\n{:25s}{:10.3f}'.format('Accuracy', accuracy)

    return score_str




def report_multiclass_scores(y, y_pred):

    hamm = hamming_loss(y, y_pred)
    f1_macro = f1_score(y, y_pred, average='macro')
    f1_micro = f1_score(y, y_pred, average='micro')
    accuracy = accuracy_score(y, y_pred)

    score_str = '\n{:25s}{:>10.3f}\n'.format('Hamming Loss:', hamm)
    score_str += '\n\n{:25s}{:10.3f}'.format('Micro-f1 score:', f1_micro)
    score_str += ('\n{:25s}{:>10.3f}'.format('Macro-f1 score:', f1_macro))

    score_str += '\n\n{:25s}{:10.3f}'.format('Accuracy', accuracy)

    return score_str


def report_mean_scores(y_accum):

    hamm_accum = []
    f1_accum = []
    f1_macro_accum = []
    f1_micro_accum = []
    accuracy_accum = []

    for y, y_pred in y_accum:
        hamm_accum.append(hamming_loss(y, y_pred))
        f1_accum.append(f1_score(y, y_pred, average=None))
        f1_macro_accum.append(f1_score(y, y_pred, average='macro'))
        f1_micro_accum.append(f1_score(y, y_pred, average='micro'))
        accuracy_accum.append(accuracy_score(y, y_pred))

    hamm = np.mean(hamm_accum)
    f1 = np.mean(f1_accum, axis=0)
    f1_macro = np.mean(f1_macro_accum)
    f1_micro = np.mean(f1_micro_accum)
    accuracy = np.mean(accuracy_accum)

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

    scores = zip(classes, f1, np.add.reduce(y))
    score_str = '\n{:25s}{:>10.3f}\n'.format('Hamming Loss:', hamm)
    score_str += '\nf1 scores \n -----------'
    for label, score, freq in sorted(scores, key=lambda s: s[0], reverse=False):
        score_str += '\n{:25s}{:>10.3f}{:>10d}'.format(label.capitalize() + ':', score, freq)
    score_str += '\n\n{:25s}{:10.3f}'.format('Micro-f1 score:', f1_micro)
    score_str += ('\n{:25s}{:>10.3f}'.format('Macro-f1 score:', f1_macro))


    score_str += '\n\n{:25s}{:10.3f}'.format('Accuracy', accuracy)

    return score_str

def report_mean_multiclass_scores(y_accum):

    hamm_accum = []
    f1_macro_accum = []
    f1_micro_accum = []
    accuracy_accum = []

    for y, y_pred in y_accum:
        hamm_accum.append(hamming_loss(y, y_pred))
        f1_macro_accum.append(f1_score(y, y_pred, average='macro'))
        f1_micro_accum.append(f1_score(y, y_pred, average='micro'))
        accuracy_accum.append(accuracy_score(y, y_pred))

    hamm = np.mean(hamm_accum)
    f1_macro = np.mean(f1_macro_accum)
    f1_micro = np.mean(f1_micro_accum)
    accuracy = np.mean(accuracy_accum)
    
    score_str = '\n{:25s}{:>10.3f}\n'.format('Hamming Loss:', hamm)
    score_str += '\n\n{:25s}{:10.3f}'.format('Micro-f1 score:', f1_micro)
    score_str += ('\n{:25s}{:>10.3f}'.format('Macro-f1 score:', f1_macro))


    score_str += '\n\n{:25s}{:10.3f}'.format('Accuracy', accuracy)

    return score_str
