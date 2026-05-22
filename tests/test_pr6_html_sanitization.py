"""
test_pr6_html_sanitization.py
==============================
PR6: HTML sanitization — _sanitize_html strips dangerous content and records counts.
"""
from __future__ import annotations

import pytest

from uma.ingest.parser import _sanitize_html


def test_script_tags_removed():
    html = "<html><body><script>evil()</script><p>safe</p></body></html>"
    text, counts = _sanitize_html(html)
    assert "evil()" not in text
    assert "safe" in text
    assert counts.get("scripts", 0) >= 1


def test_iframe_removed():
    html = "<html><body><iframe src='http://evil.com'></iframe><p>ok</p></body></html>"
    text, counts = _sanitize_html(html)
    assert "evil.com" not in text
    assert "ok" in text
    assert counts.get("iframes", 0) >= 1


def test_inline_event_handler_stripped():
    html = '<p onclick="doEvil()">Click</p>'
    text, counts = _sanitize_html(html)
    assert "doEvil" not in text
    assert "Click" in text
    assert counts.get("event_handlers", 0) >= 1


def test_javascript_href_stripped():
    html = '<a href="javascript:void(0)">link</a>'
    text, counts = _sanitize_html(html)
    assert "javascript" not in text.lower()
    assert counts.get("js_urls", 0) >= 1


def test_data_uri_stripped():
    html = '<img src="data:image/png;base64,abc123">'
    text, counts = _sanitize_html(html)
    assert "data:" not in text
    assert counts.get("data_urls", 0) >= 1


def test_conditional_comment_stripped():
    html = "before <!--[if IE]><script>ie()</script><![endif]--> after"
    text, counts = _sanitize_html(html)
    assert "ie()" not in text
    assert "before" in text
    assert "after" in text
    assert counts.get("cond_comments", 0) >= 1


def test_clean_html_produces_zero_counts():
    html = "<html><body><p>Hello world</p><ul><li>item</li></ul></body></html>"
    text, counts = _sanitize_html(html)
    assert "Hello world" in text
    assert counts == {}


def test_noscript_removed():
    html = "<html><body><noscript>enable js</noscript><p>content</p></body></html>"
    text, counts = _sanitize_html(html)
    assert "enable js" not in text
    assert "content" in text


def test_object_embed_removed():
    html = "<html><body><object data='x.swf'></object><embed src='y.swf'><p>text</p></body></html>"
    text, counts = _sanitize_html(html)
    assert "x.swf" not in text
    assert "text" in text


def test_malicious_fixture_file():
    import os
    fixture = os.path.join(os.path.dirname(__file__), "fixtures", "security", "malicious.html")
    with open(fixture, encoding="utf-8") as fh:
        html = fh.read()
    text, counts = _sanitize_html(html)
    assert "alert" not in text
    assert "Safe content" in text
    assert sum(counts.values()) >= 3


def test_svg_removed():
    html = '<html><body><svg><script>evil()</script></svg><p>text</p></body></html>'
    text, counts = _sanitize_html(html)
    assert "evil()" not in text
    assert "text" in text
