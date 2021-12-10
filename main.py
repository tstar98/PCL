import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


# helper function to save predictions to an output file
def labels2file(p, outf_path):
    with open(outf_path, 'w') as outf:
        for pi in p:
            outf.write('\t'.join([str(k) for k in pi]) + '\n')


def run():
    # get training data
    train = pd.read_csv("pcl_train.tsv", sep='\t')
    test = pd.read_csv("pcl_test.tsv", sep='\t')

    # vectorize text
    vectorizer = CountVectorizer(stop_words="english")
    x_train = vectorizer.fit_transform(train["text"].values.astype("U"))
    y_train = [0 if label in (0, 1) else 1 for label in train["label"].values]

    # train the model
    print("Training...")
    model = MLPClassifier(hidden_layer_sizes=(20,), solver='lbfgs')
    model.fit(x_train, y_train)

    # get testing data
    x_test = vectorizer.transform(test["text"].values.astype("U"))
    y_test = [0 if label in (0, 1) else 1 for label in test["label"].values]

    # testing
    print("Testing...")
    y_pred = model.predict(x_test)

    file = open("output", "a")
    for i in range(len(y_pred)):
        file.write(f"{test['par_id'].values[i]}\t{test['art_id'].values[i]}\t{test['keyword'].values[i]}"
                   f"\t{test['country_code'].values[i]}\t\t{test['text'].values[i]}{y_pred[i]}\n")

    print(f"Precision: {precision_score(y_test, y_pred)}")
    print(f"Recall: {recall_score(y_test, y_pred)}")
    print(f"F score: {f1_score(y_test, y_pred)}")


def split_data():
    file = open("dev_set.csv")
    par_id = {}
    for line in file:
        line = line.split(",")
        par_id[line[0]] = True

    file = open("pcl.tsv")
    file1 = open("pcl_train.tsv", "a")
    file2 = open("pcl_test.tsv", "a")

    for line in file:
        data = line.split('\t')
        if data[0] in par_id:
            file2.write(f"{line}")
        else:
            file1.write(f"{line}")


if __name__ == "__main__":
    run()
