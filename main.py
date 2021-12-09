import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


# helper function to save predictions to an output file
def labels2file(p, outf_path):
    with open(outf_path, 'w') as outf:
        for pi in p:
            outf.write('\t'.join([str(k) for k in pi]) + '\n')


if __name__ == "__main__":
    # get training data
    df = pd.read_csv("pcl.tsv", sep='\t')

    train, test = train_test_split(df, test_size=0.2)

    # vectorize text
    vectorizer = CountVectorizer(stop_words="english")
    x_train = vectorizer.fit_transform(train["text"].values.astype("U"))
    y_train = [0 if label in (0, 1) else 1 for label in train["label"].values]

    # train the model
    model = SGDClassifier()
    model.fit(x_train, y_train)

    # get testing data
    x_test = vectorizer.transform(test["text"].values.astype("U"))
    y_test = [0 if label in (0, 1) else 1 for label in test["label"].values]

    # testing
    y_pred = model.predict(x_test)
    pred = []
    #
    # for i in range(len(y_pred)):
    #     pred.append([df1["par_id"].values[i], art_id_test[i], df1["keyword"].values[i],
    #                  df1["country_code"].values[i], df1["text"].values[i], y_pred[i]])
    #
    # labels2file(pred, "output.txt")

    print(f"Precision: {precision_score(y_test, y_pred)}")
    print(f"Recall: {recall_score(y_test, y_pred)}")
    print(f"F score: {f1_score(y_test, y_pred)}")
