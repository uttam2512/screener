import docx
import spacy
#from rake_new import Rake  # ✅ use rake-new instead

nlp = spacy.load("en_core_web_sm")

def extract_text_from_docx(path):
    doc = docx.Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_keywords(text):
    r = Rake()
    r.extract_keywords_from_text(text)
    return r.get_ranked_phrases()

def get_cleaned_keywords(text):
    doc = nlp(text)
    keywords = set()
    for chunk in doc.noun_chunks:
        keywords.add(chunk.text.lower().strip())
    return list(keywords)

import re
from collections import defaultdict

def extract_keywords(text):
    # Very basic stopwords
    stop_words = set("""
        a about above after again against all am an and any are aren't as at be because been before being
        below between both but by can't cannot could couldn't did didn't do does doesn't doing don't down during each
        few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers
        herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me
        more most mustn't my myself no nor not of off on once only or other ought our ours ourselves out over
        own same shan't she she'd she'll she's should shouldn't so some such than that that's the their theirs them
        themselves then there there's these they they'd they'll they're they've this those through to too under until up very
        was wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while who who's whom
        why why's with won't would wouldn't you you'd you'll you're you've your yours yourself yourselves
    """.split())

    # Tokenize and filter
    words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
    freq = defaultdict(int)
    for word in words:
        if word not in stop_words:
            freq[word] += 1

    # Sort by frequency
    sorted_keywords = sorted(freq.items(), key=lambda x: -x[1])
    return [word for word, _ in sorted_keywords[:20]]


if __name__ == "__main__":
    path = "/Users/uttam/Desktop/screener/job_descriptions/sample_jd.docx"
    text = extract_text_from_docx(path)
    keywords = extract_keywords(text)
    print("RAKE:", keywords[:10])
    print("spaCy noun phrases:", get_cleaned_keywords(text)[:10])
