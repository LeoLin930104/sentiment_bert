print("Importing Dependencies")
import re
import pandas as pd

# ---------- Configurable patterns ----------
URL_RE          = re.compile(r'https?://\S+')
TRANSLATE_RE    = re.compile(r'Translate post')
USERNAME_RE     = re.compile(r'^\s*.*?@\w+\s+(?:Follow)?', flags=re.UNICODE)
TIMESTAMP_RE    = re.compile(
    r'\s*(?:午前|午後)?\d{1,2}:\d{2}\s*(?:AM|PM)?'     # clock
    r'\s*・?\s*'                                       # optional dot
    r'(?:\d{4}[年/-]\d{1,2}[月/-]\d{1,2}日?)?',         # date (JP or ISOish)
    flags=re.UNICODE)
ENGLISH_RE      = re.compile(r'[A-Za-z]+')
WS_RE           = re.compile(r'\s+')
TRAILING_NOISE_RE = re.compile(r'[\d.,\s?@_]+$')

# ---------- Japanese character definitions ----------
JP_RANGE   = r'\u3040-\u30FF\u4E00-\u9FFF'
LAST_JP_RE = re.compile(rf'^(.*[{JP_RANGE}]).*$', re.UNICODE)
JP_CHAR_RE = re.compile(rf'[{JP_RANGE}]')

# ---------- Allowed punctuation ----------
ALLOWED_PUNCT = r'()/,.!?+-:＆「」。、！？()％%#＃'
KEEP_RE = re.compile(
    rf'[^\u3040-\u30FF\u4E00-\u9FFF0-9 {re.escape(ALLOWED_PUNCT)}]',
    flags=re.UNICODE
)

# ---------- Blocklist of phrases to exclude tweets ----------
BLOCKLIST_PHRASES = [
    '≪緊急事態宣言発令中≫', '新型コロナウイルス', '緊急事態宣言', '感染予防対策', '新型肺炎', 'ワクチン'
]

# ---------- Cleaning function ----------
def clean(text: str) -> str | None:
    if not isinstance(text, str):
        return None

    # Filter out tweets with blocked phrases
    for phrase in BLOCKLIST_PHRASES:
        if phrase in text:
            return None

    # Preserve special phrases
    text = text.replace('PFAS', '__PFAS__').replace('pfas', '__pfas__')

    # Remove known artifacts
    text = URL_RE.sub('', text)
    text = TRANSLATE_RE.sub('', text)
    text = USERNAME_RE.sub('', text)
    text = TIMESTAMP_RE.sub('', text)
    text = ENGLISH_RE.sub('', text)
    text = TRAILING_NOISE_RE.sub('', text)

    # Restore special phrases
    text = text.replace('__PFAS__', 'PFAS').replace('__pfas__', 'pfas')

    # Truncate at last Japanese character
    m = LAST_JP_RE.match(text)
    if m:
        text = m.group(1)

    # Remove disallowed characters
    text = KEEP_RE.sub('', text)

    # Normalize whitespace
    text = WS_RE.sub(' ', text).strip()

    # Filter out tweets with < 90% Japanese content
    if len(text) == 0:
        return None
    jp_chars = len(JP_CHAR_RE.findall(text))
    if jp_chars / len(text) < 0.9:
        return None

    return text

# ---------- File processing ----------
if __name__ == '__main__':
    usernames = [
        "ecoyuri",
        "shinji_ishimaru",
        "renho_sha",
        "toshio_tamogami",
    ]
    path = 'data/tweets/'

    for username in usernames:
        input = f"{path}raw/{username}_tweets.csv"
        output = f"{path}preprocessed/{username}_tweets.csv"

        df = pd.read_csv(input)
        if 'tweet' not in df.columns:
            raise ValueError(f'"tweet" column not found in {input}')

        df['tweet'] = df['tweet'].map(clean)
        df = df[df['tweet'].notnull()]

        df.to_csv(output, index=False)
        print(f'✓ Saved {output}')
