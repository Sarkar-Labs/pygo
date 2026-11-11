"""PyGo Tokenizer — BPE / WordPiece / SentencePiece-style tokenization."""
from __future__ import annotations
import re
import json
import unicodedata
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple, Set


# ─── Base Tokenizer ───────────────────────────────────────────────────────────

class Tokenizer:
    def encode(self, text: str) -> List[int]: raise NotImplementedError
    def decode(self, ids: List[int]) -> str: raise NotImplementedError
    def tokenize(self, text: str) -> List[str]: raise NotImplementedError

    @property
    def vocab_size(self) -> int: raise NotImplementedError


# ─── Byte-Pair Encoding (GPT-2 / GPT-3 / GPT-4 style) ─────────────────────────

class BPETokenizer(Tokenizer):
    """
    Byte-level BPE: operates on UTF-8 bytes (256 base tokens), so it can
    represent any Unicode string with no <UNK> tokens, matching GPT-2/3/4.
    """

    def __init__(self, vocab: Dict[str, int] = None, merges: List[Tuple[str,str]] = None):
        self.encoder = vocab or {}
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges or [])}
        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.cache: Dict[str, str] = {}

        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            re.UNICODE
        ) if False else re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\sa-zA-Z0-9]+|\s+(?!\S)|\s+"""
        )

        self.special_tokens: Dict[str, int] = {}

    @staticmethod
    def _bytes_to_unicode() -> Dict[int, str]:
        """Maps every byte value to a printable unicode char (avoids
        whitespace/control chars breaking BPE merge regex)."""
        bs = (list(range(ord("!"), ord("~") + 1)) +
              list(range(ord("¡"), ord("¬") + 1)) +
              list(range(ord("®"), ord("ÿ") + 1)))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return dict(zip(bs, (chr(c) for c in cs)))

    def add_special_tokens(self, tokens: List[str]):
        for tok in tokens:
            if tok not in self.encoder:
                idx = len(self.encoder)
                self.encoder[tok] = idx
                self.decoder[idx] = tok
                self.special_tokens[tok] = idx

    def _get_pairs(self, word: Tuple[str, ...]) -> Set[Tuple[str,str]]:
        pairs = set()
        prev = word[0]
        for c in word[1:]:
            pairs.add((prev, c))
            prev = c
        return pairs

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]

        word = tuple(token)
        pairs = self._get_pairs(word)
        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if i < len(word) - 1 and word[i] == first and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = self._get_pairs(word)

        result = " ".join(word)
        self.cache[token] = result
        return result

    def tokenize(self, text: str) -> List[str]:
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token_bytes = token.encode("utf-8")
            mapped = "".join(self.byte_encoder[b] for b in token_bytes)
            bpe_tokens.extend(self.bpe(mapped).split(" "))
        return bpe_tokens

    def encode(self, text: str) -> List[int]:
        return [self.encoder.get(t, self.encoder.get("<unk>", 0))
                for t in self.tokenize(text)]

    def decode(self, ids: List[int]) -> str:
        text = "".join(self.decoder.get(i, "") for i in ids)
        byte_array = bytearray(self.byte_decoder[c] for c in text
                               if c in self.byte_decoder)
        return byte_array.decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    @classmethod
    def train(cls, corpus: List[str], vocab_size: int = 32000,
             special_tokens: Optional[List[str]] = None) -> "BPETokenizer":
        """Train BPE merges from a text corpus."""
        tok = cls()
        byte_enc = tok._bytes_to_unicode()

        # Initial vocab: 256 byte-level tokens
        vocab = {chr(v): k for k, v in byte_enc.items()}
        word_freqs: Counter = Counter()

        pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\sa-zA-Z0-9]+|\s+(?!\S)|\s+""")
        for line in corpus:
            for word in re.findall(pat, line):
                mapped = "".join(byte_enc[b] for b in word.encode("utf-8"))
                word_freqs[mapped] += 1

        splits = {w: tuple(w) for w in word_freqs}
        merges: List[Tuple[str,str]] = []
        target_merges = vocab_size - 256 - len(special_tokens or [])

        for _ in range(target_merges):
            pair_freqs: Counter = Counter()
            for word, freq in word_freqs.items():
                split = splits[word]
                for i in range(len(split) - 1):
                    pair_freqs[(split[i], split[i+1])] += freq

            if not pair_freqs:
                break
            best_pair = max(pair_freqs, key=pair_freqs.get)
            merges.append(best_pair)

            new_token = best_pair[0] + best_pair[1]
            vocab[new_token] = len(vocab)

            for word in splits:
                split = list(splits[word])
                new_split = []
                i = 0
                while i < len(split):
                    if (i < len(split) - 1 and
                            split[i] == best_pair[0] and split[i+1] == best_pair[1]):
                        new_split.append(new_token)
                        i += 2
                    else:
                        new_split.append(split[i])
                        i += 1
                splits[word] = tuple(new_split)

        encoder = {tok: idx for tok, idx in vocab.items()}
        tok.encoder = encoder
        tok.decoder = {v: k for k, v in encoder.items()}
        tok.bpe_ranks = {pair: i for i, pair in enumerate(merges)}

        if special_tokens:
            tok.add_special_tokens(special_tokens)
        return tok

    def save(self, vocab_path: str, merges_path: str):
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.encoder, f, ensure_ascii=False)
        with open(merges_path, "w", encoding="utf-8") as f:
            for pair in sorted(self.bpe_ranks, key=self.bpe_ranks.get):
                f.write(f"{pair[0]} {pair[1]}\n")

    @classmethod
    def load(cls, vocab_path: str, merges_path: str) -> "BPETokenizer":
        with open(vocab_path, encoding="utf-8") as f:
            vocab = json.load(f)
        merges = []
        with open(merges_path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split(" ")
                if len(parts) == 2:
                    merges.append(tuple(parts))
        return cls(vocab, merges)


# ─── WordPiece (BERT-style) ───────────────────────────────────────────────────

class WordPieceTokenizer(Tokenizer):
    def __init__(self, vocab: Dict[str, int], unk_token="[UNK]",
                max_input_chars_per_word=200):
        self.vocab = vocab
        self.ids_to_tokens = {v: k for k, v in vocab.items()}
        self.unk_token = unk_token
        self.max_chars = max_input_chars_per_word

    def tokenize(self, text: str) -> List[str]:
        output_tokens = []
        for token in self._basic_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_chars:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens

    @staticmethod
    def _basic_tokenize(text: str) -> List[str]:
        text = unicodedata.normalize("NFD", text)
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")
        text = re.sub(r"([\u4e00-\u9fff])", r" \1 ", text)  # CJK char splitting
        tokens = text.strip().split()
        out = []
        for tok in tokens:
            out.extend(re.findall(r"\w+|[^\w\s]", tok))
        return out

    def encode(self, text: str) -> List[int]:
        return [self.vocab.get(t, self.vocab.get(self.unk_token, 0))
                for t in self.tokenize(text)]

    def decode(self, ids: List[int]) -> str:
        tokens = [self.ids_to_tokens.get(i, self.unk_token) for i in ids]
        text = " ".join(tokens).replace(" ##", "")
        return text.strip()

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @classmethod
    def train(cls, corpus: List[str], vocab_size: int = 30000) -> "WordPieceTokenizer":
        """Greedy WordPiece training using likelihood-gain merge criterion."""
        word_freqs: Counter = Counter()
        for line in corpus:
            for word in line.split():
                word_freqs[word] += 1

        alphabet = set()
        for word in word_freqs:
            alphabet.add(word[0])
            for c in word[1:]:
                alphabet.add("##" + c)

        vocab = {c: i for i, c in enumerate(sorted(alphabet))}
        for special in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]:
            vocab[special] = len(vocab)

        splits = {w: ([w[0]] + ["##" + c for c in w[1:]]) for w in word_freqs}

        while len(vocab) < vocab_size:
            pair_scores = defaultdict(float)
            pair_counts = defaultdict(int)
            letter_counts = defaultdict(int)

            for word, freq in word_freqs.items():
                split = splits[word]
                if len(split) == 1:
                    letter_counts[split[0]] += freq
                    continue
                for i in range(len(split) - 1):
                    pair_counts[(split[i], split[i+1])] += freq
                    letter_counts[split[i]] += freq
                letter_counts[split[-1]] += freq

            for pair, count in pair_counts.items():
                score = count / (letter_counts[pair[0]] * letter_counts[pair[1]] + 1e-9)
                pair_scores[pair] = score

            if not pair_scores:
                break
            best_pair = max(pair_scores, key=pair_scores.get)
            a, b = best_pair
            merged = a + (b[2:] if b.startswith("##") else b)
            vocab[merged] = len(vocab)

            for word in splits:
                split = splits[word]
                new_split = []
                i = 0
                while i < len(split):
                    if i < len(split)-1 and split[i] == a and split[i+1] == b:
                        new_split.append(merged)
                        i += 2
                    else:
                        new_split.append(split[i])
                        i += 1
                splits[word] = new_split

        return cls(vocab)
