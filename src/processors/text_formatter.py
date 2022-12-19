import re
import string
import spacy
from spacy.matcher import Matcher


class TextPreProcessor:

    def __init__(self, regex_patterns: list, entities: list, patterns_dict: dict):
        self.regex_patterns = regex_patterns
        self.entities = entities
        self.patterns_dict = patterns_dict
        self.nlp = spacy.load("pt_core_news_lg")
        self.nlp.add_pipe("emoji", first=True)

    def strip_links(self, text: str) -> str:
        for pattern in self.regex_patterns:
            text = re.sub(pattern, '', text)
        return " ".join(text.split())

    def strip_all_entities(self, text: str) -> str:
        entity_prefixes = self.entities
        words = []
        for word in text.split():
            word = word.strip()
            if word:
                if word[0] not in entity_prefixes:
                    words.append(word)
        return " ".join(words)

    def remove_punctuation(self, text: str) -> str:
        entity_prefixes = self.entities
        for separator in string.punctuation:
            if separator not in entity_prefixes:
                text = text.replace(separator, ' ')
        return " ".join(text.split())

    def remove_stopwords(self, text: str) -> str:
        text = self.nlp(text)
        text = [word.text for word in text if not word.is_stop]
        return " ".join(text)

    def lemmatization(self, text: str) -> str:
        text = self.nlp(text)
        text = [word.lemma_ for word in text]
        return " ".join(text)

    def remove_emoji(self, text: str) -> str:
        text = self.nlp(text)
        text = [word.text for word in text if not word._.is_emoji]
        return " ".join(text)

    def replace_matches(self, text: str) -> list[str]:
        matcher = Matcher(self.nlp.vocab)
        doc = self.nlp(text)
        for key, value in self.patterns_dict.items():
            matcher.add(key, value)
        parsed_doc = doc.text
        for match_id, start, end in matcher(doc):
            string_id = self.nlp.vocab.strings[match_id]
            span = doc[start:end]
            parsed_doc = parsed_doc.replace(span.text, string_id)

        return parsed_doc
