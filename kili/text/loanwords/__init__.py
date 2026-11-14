"""English loanwords in Tamil script -> Latin + hyphenated Tamil suffix (`மீட்டிங்க்கு` -> `meeting-க்கு`).

Runtime engine (engine.py) and its table (loans.tsv). See engine.py for the rules and the table format.
"""
from kili.text.loanwords.engine import Latinizer, english_to_tamil, latinize

__all__ = ["Latinizer", "english_to_tamil", "latinize"]
