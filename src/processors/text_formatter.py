import re
from typing import List, Dict, Any
import spacy
from spacy.matcher import Matcher
import nltk


class TextPreProcessor:
    nlp_object = spacy.load("pt_core_news_lg", disable=["parser", "ner", "tagger"])
    nlp_object.add_pipe("emoji", first=True)

    @classmethod
    def remove_with_regex_corpus(cls, corpus: List[str], regex_patterns: List[str]) -> List[str]:
        formatted_corpus = []
        for text in corpus:
            formatted_corpus.append(cls.__remove_with_regex(text, regex_patterns))
        return formatted_corpus

    @classmethod
    def remove_with_prefixes_corpus(cls, corpus: List[str], prefixes: List[str]) -> List[str]:
        formatted_corpus = []
        for text in corpus:
            formatted_corpus.append(cls.__remove_with_prefixes(text, prefixes))
        return formatted_corpus

    @classmethod
    def remove_punctuation_corpus(cls, corpus: List[str]) -> List[str]:
        formatted_corpus = []
        for text in corpus:
            formatted_corpus.append(cls.__remove_punctuation(text))
        return formatted_corpus

    @classmethod
    def lower_text_corpus(cls, corpus: List[str]) -> List[str]:
        formatted_corpus = []
        for text in corpus:
            formatted_corpus.append(cls.__lower_text(text))
        return formatted_corpus

    @classmethod
    def remove_stopwords_corpus(cls, corpus: List[str]) -> List[str]:
        formatted_corpus = []
        for text in corpus:
            formatted_corpus.append(cls.__remove_stopwords(text))
        return formatted_corpus

    @classmethod
    def remove_emojis_corpus(cls, corpus: List[str]) -> List[str]:
        formatted_corpus = []
        for text in corpus:
            formatted_corpus.append(cls.__remove_emojis(text))
        return formatted_corpus

    @classmethod
    def lemmatization_corpus(cls, corpus: List[str]) -> List[str]:
        formatted_corpus = []
        for text in corpus:
            formatted_corpus.append(cls.__lemmatization(text))
        return formatted_corpus

    @classmethod
    def steaming_corpus(cls, corpus: List[str]) -> List[str]:
        formatted_corpus = []
        for text in corpus:
            formatted_corpus.append(cls.__steaming(text))
        return formatted_corpus

    @classmethod
    def replace_matches_corpus(cls, corpus: List[str],
                               patterns_dict: Dict[str, List[List[Dict[str, Any]]]]) -> List[str]:
        formatted_corpus = []
        for text in corpus:
            formatted_corpus.append(cls.__replace_matches(text, patterns_dict))
        return formatted_corpus

    @staticmethod
    def __remove_with_regex(raw_text: str, regex_patterns: List[str]) -> str:
        formatted_text = raw_text
        for pattern in regex_patterns:
            formatted_text = re.sub(pattern, '', formatted_text)
        return " ".join(formatted_text.split())

    @staticmethod
    def __remove_with_prefixes(raw_text: str, prefixes: List[str]) -> str:
        words = []
        for word in raw_text.split():
            word = word.strip()
            if word:
                if word[0] not in prefixes:
                    words.append(word)
        return " ".join(words)

    @classmethod
    def __remove_punctuation(cls, raw_text: str) -> str:
        doc = cls.nlp_object(raw_text)
        formatted_text = " ".join([token.text for token in doc if not token.is_punct])
        return formatted_text

    @staticmethod
    def __lower_text(raw_text: str) -> str:
        return raw_text.lower()

    @classmethod
    def __remove_stopwords(cls, raw_text: str) -> str:
        doc = cls.nlp_object(raw_text)
        formatted_text = [token.text for token in doc if not token.is_stop]
        return " ".join(formatted_text)

    @classmethod
    def __remove_emojis(cls, raw_text: str) -> str:
        doc = cls.nlp_object(raw_text)
        formatted_text = [token.text for token in doc if not token._.is_emoji]
        return " ".join(formatted_text)

    @classmethod
    def __lemmatization(cls, raw_text: str) -> str:
        doc = cls.nlp_object(raw_text)
        formatted_text = [token.lemma_ for token in doc]
        return " ".join(formatted_text)

    @staticmethod
    def __steaming(raw_text: str) -> str:
        stemmer = nltk.stem.RSLPStemmer()
        formatted_text = [stemmer.stem(token) for token in raw_text.split()]
        return " ".join(formatted_text)

    @classmethod
    def __replace_matches(cls, raw_text: str, patterns_dict: Dict[str, List[List[Dict[str, Any]]]]) -> str:
        matcher = Matcher(cls.nlp_object.vocab)
        doc = cls.nlp_object(raw_text)
        for key, value in patterns_dict.items():
            matcher.add(key, value)
        parsed_doc = doc.text
        for match_id, start, end in matcher(doc):
            string_id = cls.nlp_object.vocab.strings[match_id]
            span = doc[start:end]
            parsed_doc = parsed_doc.replace(span.text, string_id)

        return parsed_doc
