"""Explicit text variants shared by native selectors and local OCR."""
import re
import unicodedata


def within_edits(a, b, limit):
    if abs(len(a) - len(b)) > limit:
        return False
    previous = list(range(len(b) + 1))
    for i, char in enumerate(a, 1):
        row = [i] + [limit + 1] * len(b)
        for j in range(max(1, i-limit), min(len(b), i+limit) + 1):
            row[j] = min(row[j-1] + 1, previous[j] + 1,
                         previous[j-1] + (char != b[j-1]))
        if min(row) > limit:
            return False
        previous = row
    return previous[-1] <= limit


class TextPattern:
    """Aliases or a full-match regex; fuzzy aliases are explicit and bounded."""

    def __init__(self, values=(), *, regex=None, max_edits=0, ignore_case=True,
                 ignore_space=True, normalize_unicode=True):
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, (list, tuple)) or any(not isinstance(v, str) for v in values):
            raise ValueError("Text aliases must be a string or list of strings.")
        if bool(values) == bool(regex):
            raise ValueError("Specify aliases or a regex, not both.")
        if type(max_edits) is not int or not 0 <= max_edits <= 2 or (regex and max_edits):
            raise ValueError("Aliases support zero, one or two character edits.")
        self.ignore_case, self.ignore_space = ignore_case, ignore_space
        self.normalize_unicode, self.max_edits = normalize_unicode, max_edits
        self.values = tuple(dict.fromkeys(self.normalize(v) for v in values))
        if any(not v for v in self.values):
            raise ValueError("Text aliases cannot be empty.")
        if max_edits and any(len(v) < max(6, 5*max_edits) for v in self.values):
            raise ValueError("Fuzzy aliases need at least 6 characters (10 for two edits).")
        self.regex = re.compile(regex, re.IGNORECASE if ignore_case else 0) if regex else None
        if self.regex and self.regex.fullmatch(""):
            raise ValueError("A text regex must not match an empty label.")

    def normalize(self, value):
        if self.normalize_unicode:
            value = unicodedata.normalize("NFKC", value)
        if self.ignore_case:
            value = value.casefold()
        return "".join(value.split()) if self.ignore_space else value

    def matches(self, value, exact=True):
        value = self.normalize(value)
        if self.regex:
            return bool(self.regex.fullmatch(value))
        if not exact:
            if self.max_edits:
                raise ValueError("Fuzzy text matching must match the whole label.")
            return any(alias in value for alias in self.values)
        return value in self.values or bool(self.max_edits and any(
            within_edits(alias, value, self.max_edits) for alias in self.values))
