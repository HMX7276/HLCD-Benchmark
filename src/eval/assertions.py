"""Conservative English assertion scopes for lexical metrics, not a clinical NLP gold standard."""
import re


def entity_text(item):
    if not isinstance(item, dict):
        return str(item)
    text = str(item.get('name', ''))
    prefix = {'absent':'No evidence of ', 'negated':'No evidence of ',
              'historical':'History of ', 'uncertain':'Possible ',
              'family':'Family history of '}.get(item.get('assertion'), '')
    return prefix + text


def asserted_clauses(text):
    # Contrast and sentence boundaries end a scope. Negation lists remain
    # together ("no fever or chills"); commas preserve conservative scope.
    clauses = re.split(r'[;.!?\n]|\b(?:but|however|although)\b', str(text), flags=re.I)
    result = []
    for clause in clauses:
        c = clause.strip().lower()
        if not c:
            continue
        c = re.sub(r'\bnot only\b', '', c)
        if re.search(r'\bfamily history\b', c):
            state = 'family'
        elif re.search(r'\b(?:cannot (?:rule out|exclude)|rule out|possible|possibly|suspect(?:ed)?|probable|concern for)\b', c):
            state = 'uncertain'
        elif re.search(r'\b(?:no(?: evidence of)?|not|without|denies|denied|negative for|ruled out|absent)\b', c):
            state = 'absent'
        elif re.search(r'\b(?:history of|previous|prior|resolved)\b', c):
            state = 'historical'
        else:
            state = 'present'
        # Remove cues before lexical comparison; matching only a shared cue
        # must not count as a shared clinical entity.
        body = re.sub(r'\b(?:family history of|history of|no evidence of|negative for|cannot rule out|cannot exclude|rule out|ruled out|concern for|no|not|without|denies|denied|absent|possible|possibly|suspected|suspect|probable|previous|prior|resolved)\b', '', c)
        if body.strip():
            result.append((state, body.strip()))
    return result
