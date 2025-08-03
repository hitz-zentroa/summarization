from dataclasses import dataclass

from lingua import Language, LanguageDetectorBuilder
from nltk import tokenize

from src.constants import Criterion

# TODO: loaded even if not necessary
languages = [Language.ENGLISH, Language.SPANISH, Language.BASQUE]
detector = LanguageDetectorBuilder.from_languages(*languages).build()


@dataclass
class Document:
    idx: str
    round: int
    original_document: str
    reference_summaries: list[str]
    model_summaries: dict[str, 'Summary']
    _original_document_tokens: list[str] = None
    _original_document_vocab: set[str] = None
    _reference_summary_tokens: list[list[str]] = None
    _reference_summary_vocab: set[str] = None

    @classmethod
    def from_json(cls, data: dict):
        for k, v in data['model_summaries'].items():
            data['model_summaries'][k] = Summary.from_json(v)
        return cls(**data)

    def to_json(self):
        return dict(
            idx=self.idx,
            round=self.round,
            original_document=self.original_document,
            reference_summaries=self.reference_summaries,
            model_summaries={k: v.to_json() for k, v in self.model_summaries.items()}
        )

    @property
    def original_document_tokens(self):
        if self._original_document_tokens is None:
            self._original_document_tokens = tokenize.word_tokenize(self.original_document.lower())
        return self._original_document_tokens

    @property
    def original_document_vocab(self):
        if self._original_document_vocab is None:
            self._original_document_vocab = set(self.original_document_tokens)
        return self._original_document_vocab

    @property
    def reference_summary_tokens(self):
        if self._reference_summary_tokens is None:
            self._reference_summary_tokens = [tokenize.word_tokenize(x.lower()) for x in self.reference_summaries]
        return self._reference_summary_tokens

    @property
    def reference_summary_vocab(self):
        if self._reference_summary_vocab is None:
            self._reference_summary_vocab = set(y for x in self.reference_summary_tokens for y in x)
        return self._reference_summary_vocab


@dataclass
class Summary:
    summ: str
    anns: dict[Criterion, list[float]]
    lang: str = None
    _tokens: list[str] = None
    _vocab: set[str] = None

    @classmethod
    def from_json(cls, data: dict):
        if 'lang' not in data:
            data['lang'] = detector.detect_language_of(data['summ']).iso_code_639_1.name.lower()
        return cls(**data)

    def to_json(self):
        return dict(summ=self.summ, anns=self.anns)

    @property
    def tokens(self):
        if self._tokens is None:
            self._tokens = tokenize.word_tokenize(self.summ.lower())
        return self._tokens

    @property
    def vocab(self):
        if self._vocab is None:
            self._vocab = set(self.tokens)
        return self._vocab
