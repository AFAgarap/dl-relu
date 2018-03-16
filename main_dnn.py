from models.dnn_keras import DNN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from utils import clean_text
from utils import load_labels


def main():

    labels_path = \
        '/home/darth/GitHub Projects/senti-internship-notes/abienagarap/sentiment-analysis/dataset/tagalog/labels.txt'
    features_path = \
        '/home/darth/GitHub Projects/senti-internship-notes/abienagarap/sentiment-analysis/dataset/tagalog/tweets.txt'

    labels = load_labels(labels_path=labels_path, one_hot=True)

    features = clean_text(features_path)

    vec = TfidfVectorizer(max_features=30)
    tfidfmatrix = vec.fit_transform(features).toarray()

    train_features, test_features, train_labels, test_labels = train_test_split(tfidfmatrix, labels, test_size=0.20,
                                                                                stratify=labels)

    model = DNN(activation='swish',
                batch_size=256,
                dropout_rate=0.50,
                num_neurons=[512, 256, 128, 64, 32],
                num_features=int(tfidfmatrix.shape[1]),
                num_classes=int(labels.shape[1]),
                loss='mse',
                optimizer='adam',
                penalty_parameter=5)

    model.train(batch_size=256,
                epochs=32,
                log_path='./logs',
                n_splits=10,
                train_features=train_features,
                train_labels=train_labels,
                validation_split=0.05,
                verbose=0)

    report, confusion_matrix = model.evaluate(batch_size=512,
                                              class_names=['(0) negative', '(1) neutral', '(2) positive'],
                                              test_features=test_features,
                                              test_labels=test_labels)

    print(report)
    print(confusion_matrix)


if __name__ == '__main__':
    main()
