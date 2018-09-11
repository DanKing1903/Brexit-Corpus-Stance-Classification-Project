from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from src.features.feature_transformers import Selector, SentenceFeatures, HapaxLegomera, MyWordSequencer, WordTokenizer
from src.features.target_transformers import MyMultiLabelBinarizer, MultiTaskSplitter, LabelTransformer, MultiLabelJoiner, MyLabelEncoder



def get_feature_pipe(config, scaling=False):
    if config == 'engineered':
        sent_features = Pipeline([
            ('select', Selector(key='Utterance')),
            ('extract', SentenceFeatures()),
            ('vectorize', DictVectorizer())])

        hapax = Pipeline([
            ('select', Selector(key='Utterance')),
            ('extract', HapaxLegomera()),
            ('vectorize', DictVectorizer())])

        CV = Pipeline([
            ('select', Selector(key='Utterance')),
            ('cv', CountVectorizer(ngram_range=(0, 2)))])




        if scaling is True:
            feat_pipe = Pipeline([
                ('union', FeatureUnion(
                    transformer_list=[
                        ('features', sent_features),
                        ('hapax', hapax),
                        ('Ngrams', CV)])),
                ('scaler', StandardScaler(with_mean=False))])
        else:
            feat_pipe = Pipeline([
                ('union', FeatureUnion(
                    transformer_list=[
                        ('features', sent_features),
                        ('hapax', hapax),
                        ('Ngrams', CV)]))])


    elif config == 'linguistic':
        sent_features = Pipeline([
            ('select', Selector(key='Utterance')),
            ('extract', SentenceFeatures()),
            ('vectorize', DictVectorizer())])

        hapax = Pipeline([
            ('select', Selector(key='Utterance')),
            ('extract', HapaxLegomera()),
            ('vectorize', DictVectorizer())])

        feat_pipe = Pipeline([
            ('union', FeatureUnion(
                transformer_list=[
                    ('features', sent_features),
                    ('hapax', hapax)])),
            ('scaler', StandardScaler(with_mean=False))])


    elif config == 'embeddings':
        feat_pipe = Pipeline([
            ('select', Selector(key='Utterance')),
            ('sequence', WordTokenizer())])

    else:
        raise ValueError("Incorrect feature pipe config {}".format(config))

    return feat_pipe


def get_label_pipe(config):
    print(config)

    if config == 'multi-label':
        lab_pipe = Pipeline([
            ('transform', LabelTransformer()),
            ('binarize', MyMultiLabelBinarizer())])

    elif config == 'multi-class':
        lab_pipe = Pipeline([
            ('lt', LabelTransformer()),
            ('MLJ', MultiLabelJoiner()),
            ('MLE', MyLabelEncoder())])

    elif config == 'multi-task':
        lab_pipe = Pipeline([
            ('transform', LabelTransformer()),
            ('binarize', MyMultiLabelBinarizer()),
            ('split', MultiTaskSplitter())])
    else:
        raise ValueError("Incorrect label pipe config")


    return lab_pipe
