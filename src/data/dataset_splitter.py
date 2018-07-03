import pandas as pd


def get_train_test():
    file = "../data/raw/brexit_blog_corpus.xlsx"
    df = pd.read_excel(file, usecols="A:K")
    test = df.groupby('Stance category', group_keys=False).apply(lambda g: g.sample(n=round(0.2 * len(g)), random_state=42))
    test.sort_index(inplace=True)
    mask = df.index.isin(test.index)
    train = df[~mask]
    train.sort_index(inplace=True)

    return train, test


train, test = get_train_test()

train.to_csv("../data/raw/train_set.csv", index=False)

test.to_csv("../data/raw/test_set.csv", index=False)
