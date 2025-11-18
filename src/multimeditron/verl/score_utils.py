import re
import dataclasses
from typing import List, Union, Optional

MARKDOWN_HEADER_RE = re.compile(r"^(#{1,6})\s*(.*)(\n|^)", re.MULTILINE)
MARKDOWN_BOLD_RE = re.compile(r"\*\*([^\*]+?)\*\*")
MARKDOWN_ITALIC_RE = re.compile(r"([^\*]|^)\*([^\*]+?)\*([^\*]|$)")
MARKDOWN_STRIKETHROUGH_RE = re.compile(r"~~(.+?)~~")
MARKDOWN_INLINE_CODE_RE = re.compile(r"([^`]|^)`([^\n`]+?)`([^`]|$)")
MARKDOWN_UNORDERED_LIST_RE = re.compile(r"(\n|^)\s*[-\*\+]\s*(.+?)(\n\s*[-\*\+]\s*(.+?))*(\n|$)", re.MULTILINE)
MARKDOWN_ORDERED_LIST_RE = re.compile(r"(\n|^)\s*\d+\.\s*(.+?)(\n\s*\d+\.\s*(.+?))*(\n|$)", re.MULTILINE)
MARKDOWN_CODEBLOCK_RE = re.compile(r"^```(?:\s*(\w+))?([\s\S]*?)^```$", re.MULTILINE)
MARKDOWN_LINKS_RE = re.compile(r"(^|[^!])\[(.*?)\]\((.*?)\s?(?:\"(.*?)\")?\)")
MARKDOWN_IMAGE_RE = re.compile(r"!\[(.*?)\]\((.*?)\s?(?:\"(.*?)\")?\)")

def clamp(x, min_=None, max_=None):
    assert min_ is None or max_ is None or min_ <= max_, "Invalid clamp range"
    if min_ is not None:
        x = max(min_, x)
    if max_ is not None:
        x = min(max_, x)
    return x

@dataclasses.dataclass
class ParsedLink:
    text: str
    url: str
    alt: Optional[str] = None

@dataclasses.dataclass
class ParsedCodeBlock:
    language: Optional[str]
    code: str

@dataclasses.dataclass
class ParsedMarkdownElement:
    headers: List[str] = dataclasses.field(default_factory=list)
    bold: List[str] = dataclasses.field(default_factory=list)
    italic: List[str] = dataclasses.field(default_factory=list)
    strikethrough: List[str] = dataclasses.field(default_factory=list)
    inline_code: List[str] = dataclasses.field(default_factory=list)
    unordered_lists: List[List[str]] = dataclasses.field(default_factory=list)
    ordered_lists: List[List[str]] = dataclasses.field(default_factory=list)
    blockquotes: List[str] = dataclasses.field(default_factory=list)
    codeblocks: List[ParsedCodeBlock] = dataclasses.field(default_factory=list)
    links: List[ParsedLink] = dataclasses.field(default_factory=list)
    images: List[ParsedLink] = dataclasses.field(default_factory=list)

def simple_markdown_parser(text: str) -> ParsedMarkdownElement:
    parsed = ParsedMarkdownElement()

    parsed.headers = [match.group(0).strip('\n ') for match in MARKDOWN_HEADER_RE.finditer(text)]
    parsed.bold = [match.group(1) or match.group(2) for match in MARKDOWN_BOLD_RE.finditer(text)]
    parsed.italic = [match.group(2) or match.group(2) for match in MARKDOWN_ITALIC_RE.finditer(text)]
    parsed.strikethrough = [match.group(1) for match in MARKDOWN_STRIKETHROUGH_RE.finditer(text)]
    parsed.inline_code = [match.group(2) for match in MARKDOWN_INLINE_CODE_RE.finditer(text)]
    
    parsed.unordered_lists = []
    for match in MARKDOWN_UNORDERED_LIST_RE.finditer(text):
        items = re.findall(r"[-\*\+][\ \r\t]+([^\n]+)", match.group(0))
        if len(items) == 0:
            continue
        parsed.unordered_lists.append([
            elem.strip() for elem in items
        ])
    
    parsed.ordered_lists = []
    for match in MARKDOWN_ORDERED_LIST_RE.finditer(text):
        items = re.findall(r"\d+\.[\ \r\t]+([^\n]+)", match.group(0))
        if len(items) == 0:
            continue
        parsed.ordered_lists.append([
            elem.strip() for elem in items
        ])
    
    parsed.codeblocks = [
        ParsedCodeBlock(match.group(1), match.group(2)) for match in MARKDOWN_CODEBLOCK_RE.finditer(text)
    ]
    parsed.links = [ParsedLink(match.group(2), match.group(3), alt=match.group(4)) for match in MARKDOWN_LINKS_RE.finditer(text)]
    parsed.images = [ParsedLink(match.group(1), match.group(2), alt=match.group(3)) for match in MARKDOWN_IMAGE_RE.finditer(text)]
    return parsed

def markdown_simple_reward(text: str) -> float:
    parsed = simple_markdown_parser(text)
    score = 0.0
    if len(parsed.headers) > 2:
        score += 0.2
    if len(parsed.bold) + len(parsed.italic) + len(parsed.inline_code) > 4:
        score += 0.2
    if len(parsed.unordered_lists) + len(parsed.ordered_lists) > 2:
        score += 0.2
    if len(parsed.headers) > 0:
        score += 0.2
    if len(parsed.images) > 0:
        score *= 0.5 # Penalize images link in markdown responses
    return clamp(score, 0.0, 1.0)

def markdown_check_references(text: str, callback: callable) -> float:
    parsed = simple_markdown_parser(text)
    total_links = len(parsed.links) + len(parsed.images)
    if total_links == 0:
        return 0.0
    valid_links = 0
    for link in parsed.links + parsed.images:
        if callback(link.url):
            valid_links += 1
    return clamp(valid_links / total_links, 0.0, 1.0)

