"""Check the local learning paths without a browser, dataset or ML runtime."""

from html.parser import HTMLParser
from pathlib import Path
import re
import unicodedata
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

import pytest


ROOT = Path(__file__).resolve().parents[1]
PEDAGOGY = ROOT / "docs/pedagogy"
PAGES = sorted([*PEDAGOGY.rglob("*.md"), *PEDAGOGY.rglob("*.html")])


class _HTMLReferences(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.links = []
        self.ids = []
        self.routes = []
        self.in_routes = False
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag == "nav":
            self.in_routes = attrs.get("aria-label") == "Parcours pédagogique"
        key = "href" if tag == "a" else "src" if tag in ("img", "script") else None
        if key and key in attrs:
            self.links.append(attrs[key])
            if self.in_routes and tag == "a":
                self.routes.append(attrs[key])

    def handle_endtag(self, tag):
        if tag == "nav":
            self.in_routes = False


def _markdown_headings(source):
    # GitHub retains accents in heading anchors; repeated headings get a suffix.
    source = re.sub(r"^```[^\n]*\n.*?^```\s*$", "", source, flags=re.M | re.S)
    counts = {}
    anchors = set()
    for heading in re.findall(r"^#{1,6} (.+)$", source, flags=re.M):
        slug = "".join(
            char for char in heading.lower()
            if char in " _-" or unicodedata.category(char)[0] in "LN"
        ).replace(" ", "-")
        count = counts.get(slug, 0)
        anchors.add(f"{slug}-{count}" if count else slug)
        counts[slug] = count + 1
    return anchors


@pytest.mark.parametrize("path", PAGES, ids=lambda p: str(p.relative_to(PEDAGOGY)))
def test_pedagogical_links_resolve_to_existing_files_and_anchors(path):
    source = path.read_text()
    links = _HTMLReferences(source).links if path.suffix == ".html" else re.findall(
        r"\]\(([^\s)]+)\)", source,
    )
    for link in links:
        url = urlsplit(link)
        if url.scheme or url.netloc:
            continue
        target = (path.parent / unquote(url.path)).resolve() if url.path else path
        assert target.is_relative_to(ROOT), (path, link)
        assert target.exists(), (path, link)
        if url.fragment:
            text = target.read_text()
            anchors = _markdown_headings(text) if target.suffix == ".md" else set(
                _HTMLReferences(text).ids,
            )
            assert unquote(url.fragment) in anchors, (path, link)


def test_readme_presents_matching_branch_sections_before_convergence():
    source = (PEDAGOGY / "README.md").read_text()
    sections = dict(re.findall(r"^## ([^\n]+)\n(.*?)(?=^## |\Z)", source, flags=re.M | re.S))
    camera, lidar = sections["Branche camera-depth"], sections["Branche LiDAR"]
    labels = ["Rôle", "Entrées", "Transformations", "Sorties", "Ressources", "État et limites"]
    for branch in (camera, lidar):
        assert re.findall(r"\*\*([^*]+)\.\*\*", branch) == labels
    names = list(sections)
    convergence = names.index("Convergence : fusion et détection 3D")
    assert names.index("Branche camera-depth") < convergence
    assert names.index("Branche LiDAR") < convergence
    assert "```mermaid" in source and "CIBLE : fusion" in source


@pytest.mark.parametrize("path", sorted((PEDAGOGY / "explorers").glob("*.html")), ids=lambda p: p.name)
def test_explorers_offer_both_branches_and_global_navigation(path):
    page = _HTMLReferences(path.read_text())
    assert page.routes == ["../README.md", "camera-depth.html", "architecture.html"]
    assert len(page.ids) == len(set(page.ids)), path


@pytest.mark.parametrize("path", sorted((PEDAGOGY / "assets").glob("*.svg")), ids=lambda p: p.name)
def test_pedagogical_figures_are_well_formed_svg(path):
    assert ET.parse(path).getroot().tag == "{http://www.w3.org/2000/svg}svg"
