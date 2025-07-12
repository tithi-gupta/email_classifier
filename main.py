
from classifier.spam_classifier import SpamClassifier

def main():
    clf = SpamClassifier("data/spam.csv")
    clf.train()
    print(clf.predict("Congratulations! You've won a free iPhone"))
    print(clf.predict("Hey, let's catch up for coffee tomorrow"))

if __name__ == "__main__":
    main()
