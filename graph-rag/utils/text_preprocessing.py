import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data files
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


def remove_stop_words_from_and_lemmatise_documents(documents):
    # Initialize stop words and lemmatizer
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    # Function to preprocess text
    def preprocess_text(sentences):
        preprocessed_sentences = []

        for sentence in sentences:
            # Tokenize the sentence
            words = word_tokenize(sentence)

            # Remove stop words and lemmatize each word
            filtered_words = [
                lemmatizer.lemmatize(word.lower())
                for word in words
                if word.lower() not in stop_words and word.isalpha()
            ]

            # Join words back to form the sentence
            preprocessed_sentence = " ".join(filtered_words)
            preprocessed_sentences.append(preprocessed_sentence)

        return preprocessed_sentences

    # Preprocess the list of sentences
    preprocessed_documents = preprocess_text(documents)
    return preprocessed_documents
