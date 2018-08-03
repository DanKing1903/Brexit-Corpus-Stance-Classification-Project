from src.features.target_transformers import MultiLabelJoiner
import pickle
import itertools as it

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
combos = []
for i in range(1, 6):
    generator = it.combinations(classes, i)  # generates all label combinations
    for g in generator:
        combos.append(sorted(list(g)))

MLJ = MultiLabelJoiner()

joined = MLJ.fit_transform(combos)
filename = 'data/interim/label_encoding_classes'
outfile = open(filename, 'wb')
pickle.dump(joined, outfile)
outfile.close()
