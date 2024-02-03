# ---------------------------------
# --------人生苦短，我用python--------
# ---------------------------------
import os
import re


def remove_comment(text):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " "
        else:
            return s
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)

def remove_space(text):
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s', ' ', text)
    text = re.sub(r'\\t', ' ', text)
    text = re.sub(r' +', ' ', text)
    return text

def remove_version(text):
    cut_index = text.find(';') + 1
    sample_text1 = 'pragma solidity'
    sample_text2 = 'import'
    test_text = text[:cut_index]
    if sample_text1 in test_text or sample_text2 in test_text:
        print(text[:cut_index])
        text = text[cut_index:]
    return text

