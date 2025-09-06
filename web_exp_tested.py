"""
functional web graph explorer tool
pure functional approach to web scraping at scale
everything is a tree in a graph of trees, discovered through structural analysis
domains are navigable objects that can be recursively explored
without presumptions about structure or data. tag-agnostic, dictionary-based,
purely functional approach using pinqy
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from typing import Dict, List, Any, Callable, Optional, Tuple
import json
import time
import re
from dataclasses import dataclass, asdict
from datetime import datetime
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import sqlite3
from queue import PriorityQueue
from pinqy import from_iterable as pinqy
from suite import _c

# configure minimal logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ExplorationConfig:
    max_depth: int = 3
    max_pages_per_domain: int = 100
    delay_between_requests: float = 1.0
    respect_robots_txt: bool = True
    max_concurrent_requests: int = 3
    user_agent: str = 'FunctionalWebExplorer/1.0'
    output_format: str = 'json'
    output_file: Optional[str] = None
    custom_headers: Dict[str, str] = None

    def __post_init__(self):
        if self.custom_headers is None: self.custom_headers = {}


# --- core functions ---

def create_session(config: ExplorationConfig) -> requests.Session:
    session = requests.Session()
    session.headers.update({'User-Agent': config.user_agent, **(config.custom_headers or {})})
    return session


def check_robots_txt(domain: str, session: requests.Session) -> Callable[[str], bool]:
    try:
        rp = RobotFileParser(f"https://{domain}/robots.txt")
        rp.read()
        return lambda url: rp.can_fetch(session.headers.get('User-Agent', '*'), url)
    except:
        return lambda url: True


def fetch_page_safe(url: str, session: requests.Session, robots_checker: Callable[[str], bool]) -> Optional[
    Tuple[str, BeautifulSoup]]:
    """atomic: safely fetch and parse a single page, returning its type."""
    if not robots_checker(url):
        logger.info(f"robots.txt blocks: {url}")
        return None
    try:
        response = session.get(url, timeout=15, allow_redirects=True)
        response.raise_for_status()
        content_type = response.headers.get('content-type', '').lower()

        if 'text/html' in content_type:
            return 'html', BeautifulSoup(response.content, 'lxml')
        elif 'xml' in content_type or 'application/xml' in content_type or 'text/xml' in content_type:
            # use the robust 'xml' parser for sitemaps
            return 'xml', BeautifulSoup(response.content, 'xml')

        logger.info(f"unsupported content type '{content_type}' at {url}")
        return None
    except Exception as e:
        logger.warning(f"failed to fetch {url}: {str(e)[:100]}")
        return None


def parse_sitemap(soup: BeautifulSoup) -> List[str]:
    """extracts all urls from a sitemap or sitemap index file."""
    # handles both <sitemap><loc>...</loc></sitemap> for index files
    # and <url><loc>...</loc></url> for standard sitemaps.
    locations = pinqy(soup.find_all('loc')) \
        .select(lambda tag: tag.get_text(strip=True)) \
        .where(lambda url: url) \
        .to.list()
    return locations


def element_to_functional_dict(element) -> Optional[Dict[str, Any]]:
    if not hasattr(element, 'name'):
        text_content = str(element).strip()
        if not text_content: return None
        return {
            'type': 'text', 'content': text_content, 'length': len(text_content),
            'has_numbers': bool(re.search(r'\d', text_content)),
            'has_currency': bool(re.search(r'[\$\€\£\¥]', text_content)),
            'word_count': len(text_content.split())
        }
    if not hasattr(element, 'attrs'): return None
    attrs = dict(element.attrs)
    text = element.get_text(strip=True)
    children = pinqy(element.children).select(element_to_functional_dict).where(lambda c: c is not None).to.list()
    return {
        'type': 'element', 'tag': element.name, 'attrs': attrs,
        'classes': attrs.get('class', []), 'id': attrs.get('id'), 'text': text,
        'text_length': len(text), 'children': children, 'children_count': len(children),
        'has_text': bool(text), 'has_children': bool(children),
        'has_links': element.name == 'a' and 'href' in attrs,
        'href': attrs.get('href') if element.name == 'a' else None, 'depth': 0
    }


def set_element_depths(tree_dict: Dict[str, Any], depth: int = 0) -> Dict[str, Any]:
    if not tree_dict: return None
    result = {**tree_dict, 'depth': depth}
    if result.get('children'):
        result['children'] = [set_element_depths(child, depth + 1) for child in result['children']]
    return result


def page_to_functional_tree(soup: BeautifulSoup, base_url: str) -> Optional[Dict[str, Any]]:
    body = soup.body if soup.body else soup
    tree = element_to_functional_dict(body)
    if not tree: return None
    tree = set_element_depths(tree)
    tree['base_url'] = base_url
    tree['title'] = soup.title.get_text(strip=True) if soup.title else ''
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    tree['meta_description'] = meta_desc.get('content', '') if meta_desc else ''
    return tree


def extract_all_elements(tree_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not tree_dict or tree_dict.get('type') != 'element': return []
    elements = [tree_dict]
    elements.extend(pinqy(tree_dict.get('children', [])).select_many(extract_all_elements).to.list())
    return elements


def create_structural_signature(elem: Dict[str, Any]) -> str:
    if elem.get('type') != 'element': return 'text'
    tag = elem.get('tag', '')
    class_sig = '.'.join(sorted(elem.get('classes', []))[:3])
    child_tags = pinqy(elem.get('children', [])).where(lambda c: c and c.get('type') == 'element').select(
        lambda c: c.get('tag', '')).set.distinct().order_by(lambda x: x).to.list()
    child_sig = ','.join(child_tags)
    has_text = 'T' if elem.get('has_text') else 'N'
    has_links = 'L' if elem.get('has_links') else 'N'
    children_count = min(elem.get('children_count', 0), 10)
    return f"{tag}|{class_sig}|{child_sig}|{has_text}{has_links}{children_count}"


def discover_repeating_patterns(tree_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    all_elements = pinqy(extract_all_elements(tree_dict))
    if not all_elements.to.any(): return []
    signature_groups = all_elements.group.group_by(create_structural_signature)
    return (pinqy(signature_groups.values()).where(lambda group: len(group) >= 3).select(lambda group: {
        'signature': create_structural_signature(group[0]), 'elements': group, 'count': len(group),
        'sample_element': group[0], 'avg_depth': pinqy(group).stats.average(lambda e: e.get('depth', 0)),
        'depth_variance': pinqy(group).stats.std_dev(lambda e: e.get('depth', 0)),
        'has_consistent_text': pinqy(group).to.all(lambda e: bool(e.get('text'))),
        'consistent_links': pinqy(group).to.count(lambda e: e.get('has_links')),
        'pattern_quality_score': len(group) * (1 + (1 / (1 + pinqy(group).stats.std_dev(lambda e: e.get('depth', 0)))))
    }).order_by_descending(lambda p: p['pattern_quality_score']).to.list())


def analyze_page_semantics(tree: Dict[str, Any]) -> Dict[str, Any]:
    full_text = " ".join(pinqy(extract_all_elements(tree)).select(lambda e: e.get('text', '')).to.list()).lower()
    product_keywords = {'price', 'cart', 'add', 'sku', 'sale', '$', '€', '£'}
    article_keywords = {'author', 'published', 'updated', 'by', 'copyright', 'comments'}
    contact_keywords = {'contact', 'address', 'phone', 'email', 'hours', 'location'}
    words = pinqy(full_text.split())
    return {
        'product_score': words.where(lambda w: w in product_keywords).set.distinct().to.count(),
        'article_score': words.where(lambda w: w in article_keywords).set.distinct().to.count(),
        'contact_score': words.where(lambda w: w in contact_keywords).set.distinct().to.count(),
    }


def classify_page_functionally(tree_dict: Dict[str, Any]) -> Dict[str, Any]:
    if not tree_dict: return {'type': 'invalid', 'confidence': 0.0, 'patterns': []}
    patterns = discover_repeating_patterns(tree_dict)
    if not patterns: return {'type': 'simple', 'confidence': 0.3, 'patterns': [],
                             'reasoning': 'no repeating patterns found'}
    dominant_pattern = patterns[0]
    count, has_links, depth_consistency = (int(dominant_pattern['count']),
                                           bool(dominant_pattern['consistent_links'] > 0),
                                           bool(dominant_pattern['depth_variance'] < 1.0))
    rules = [(lambda: count >= 10 and has_links and depth_consistency, 'list', 0.9),
             (lambda: count >= 5 and has_links, 'directory', 0.8),
             (lambda: count >= 8 and not has_links, 'content_list', 0.7),
             (lambda: count >= 3 and depth_consistency, 'structured', 0.6), (lambda: True, 'detail', 0.4)]
    page_type, confidence = pinqy(rules).select(lambda r: (r[1], r[2]) if r[0]() else None).where(
        lambda x: x is not None).to.first_or_default(default=('unknown', 0.2))
    if page_type in ['list', 'directory', 'content_list']:
        semantics = analyze_page_semantics(tree_dict)
        semantic_type = max(semantics, key=semantics.get)
        if semantics[semantic_type] > 2:
            page_type = f"{semantic_type.replace('_score', '')}_{page_type}"
            confidence = min(0.95, confidence + 0.1)
    return {'type': page_type, 'confidence': confidence, 'patterns': patterns, 'dominant_pattern': dominant_pattern,
            'reasoning': f"found {count} repeating elements with {dominant_pattern['consistent_links']} links"}


def traverse_and_extract(element: Dict[str, Any], path: str) -> Optional[str]:
    try:
        parts = path.strip('.').split('.')
        current = element
        for part in parts:
            found_child = pinqy(current.get('children', [])).where(
                lambda c: c.get('tag') == part or part in c.get('classes', [])).to.first_or_default()
            if found_child:
                current = found_child
            else:
                return None
        return current.get('text')
    except (TypeError, AttributeError):
        return None


def discover_data_fields_functional(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not elements: return []
    sample_elements = pinqy(elements).take(10)
    num_samples = sample_elements.to.count()
    if num_samples == 0: return []

    def extract_text_paths(elem: Dict[str, Any], path: str = "") -> List[Dict[str, Any]]:
        if elem.get('type') == 'text' and elem.get('content'): return [{'path': path, 'content': elem['content']}]
        if elem.get('type') != 'element': return []
        tag = elem.get('tag', '')
        classes = '.'.join(elem.get('classes', [])[:2])
        current_path = f"{path}.{tag}" + (f".{classes}" if classes else "")
        paths = [{'path': current_path, 'content': elem['text']}] if elem.get('text') and not elem.get(
            'children') else []
        paths.extend(
            pinqy(elem.get('children', [])).select_many(lambda c: extract_text_paths(c, current_path)).to.list())
        return paths

    all_paths = sample_elements.select_many(extract_text_paths).to.list()
    if not all_paths: return []
    path_groups = pinqy(all_paths).group.group_by(lambda p: p['path'])
    return (pinqy(path_groups.values()).where(lambda group: len(group) >= num_samples * 0.5).select(lambda group: {
        'field_path': group[0]['path'], 'occurrence_rate': len(group) / num_samples,
        'sample_values': pinqy(group).select(lambda p: p['content'][:100]).take(3).to.list()
    }).order_by_descending(lambda f: f['occurrence_rate']).to.list())


def process_url_functionally(url: str, session: requests.Session, robots_checker: Callable[[str], bool],
                             base_domain: str) -> Optional[Dict[str, Any]]:
    """process a single url through the complete pipeline, routing by content type."""
    fetch_result = fetch_page_safe(url, session, robots_checker)
    if not fetch_result: return None

    content_type, soup = fetch_result

    # --- xml sitemap pipeline ---
    if content_type == 'xml':
        logger.info(f"processing sitemap: {url}")
        sitemap_links = parse_sitemap(soup)
        absolute_links = pinqy(sitemap_links) \
            .select(lambda link: urljoin(url, link)) \
            .where(lambda abs_url: urlparse(abs_url).netloc == base_domain) \
            .set.distinct().to.list()
        return {
            'url': url, 'title': 'sitemap', 'classification': {'type': 'sitemap', 'confidence': 1.0},
            'extraction': {'extracted_data': [], 'links': sitemap_links},
            'child_links': absolute_links, 'processed_at': datetime.now().isoformat()
        }

    # --- html pipeline ---
    elif content_type == 'html':
        tree = page_to_functional_tree(soup, url)
        if not tree: return None
        classification = classify_page_functionally(tree)
        dominant_pattern = classification.get('dominant_pattern')
        all_elements = extract_all_elements(tree)
        extracted_data = []
        if not dominant_pattern:
            links = pinqy(all_elements).where(lambda e: e.get('has_links') and e.get('href')).select(
                lambda e: e['href']).set.distinct().to.list()
        else:
            matching_elements = [e for e in all_elements if
                                 create_structural_signature(e) == dominant_pattern.get('signature')]
            data_fields = discover_data_fields_functional(matching_elements)
            extracted_data = pinqy(matching_elements).select(lambda elem: {
                field['field_path'].split('.')[-1].replace('-', '_'): traverse_and_extract(elem, field['field_path'])
                for field in data_fields
            }).where(lambda data_item: any(data_item.values())).to.list()
            links = pinqy(matching_elements).select_many(extract_all_elements).where(
                lambda descendant: descendant.get('has_links') and descendant.get('href')
            ).select(
                lambda link_element: link_element.get('href')
            ).set.distinct().to.list()
        absolute_links = pinqy(links).select(lambda link: urljoin(url, link)).where(
            lambda abs_url: urlparse(abs_url).netloc == base_domain).set.distinct().to.list()
        return {
            'url': url, 'title': tree.get('title', ''), 'classification': classification,
            'extraction': {'extracted_data': extracted_data, 'links': links},
            'child_links': absolute_links, 'processed_at': datetime.now().isoformat()
        }

    return None  # for any other content types


def explore_domain_functionally(start_urls: List[str], config: ExplorationConfig) -> Dict[str, Any]:
    if not start_urls: return {'error': 'no start urls provided'}
    domain = urlparse(start_urls[0]).netloc
    logger.info(f"exploring domain: {domain}")
    session = create_session(config)
    robots_checker = check_robots_txt(domain, session) if config.respect_robots_txt else lambda x: True
    all_results, visited_urls, pages_processed = [], set(), 0
    url_queue = PriorityQueue()
    for url in start_urls: url_queue.put((0, url))
    while not url_queue.empty() and pages_processed < config.max_pages_per_domain:
        batch_size = min(config.max_concurrent_requests, url_queue.qsize(),
                         config.max_pages_per_domain - pages_processed)
        current_batch = []
        last_priority = 0
        for _ in range(batch_size):
            if url_queue.empty(): break
            priority, url = url_queue.get()
            last_priority = priority
            if url not in visited_urls:
                current_batch.append(url)
                visited_urls.add(url)
        if not current_batch: continue
        logger.info(f"processing batch of {len(current_batch)} urls (highest priority: {last_priority})")
        with ThreadPoolExecutor(max_workers=config.max_concurrent_requests) as executor:
            future_to_url = {executor.submit(process_url_functionally, url, session, robots_checker, domain): url for
                             url in current_batch}
            batch_results = [future.result() for future in as_completed(future_to_url) if future.result()]
        all_results.extend(batch_results)
        pages_processed += len(batch_results)
        new_links_discovered = 0
        for result in batch_results:
            page_type = result.get('classification', {}).get('type', 'unknown').split('_')[-1]
            priority_score = {'list': 1, 'directory': 1, 'content_list': 2, 'structured': 3, 'detail': 4}.get(page_type,
                                                                                                              5)
            for link in result.get('child_links', []):
                if link not in visited_urls:
                    url_queue.put((priority_score, link))
                    new_links_discovered += 1
        if new_links_discovered > 0: logger.info(f"discovered {new_links_discovered} new links")
        if config.delay_between_requests > 0: time.sleep(config.delay_between_requests)
    return {
        'domain': domain, 'total_pages_processed': pages_processed, 'total_urls_visited': len(visited_urls),
        'results': all_results, 'config_used': asdict(config), 'completed_at': datetime.now().isoformat()
    }


def analyze_exploration_results(exploration_results: Dict[str, Any]) -> Dict[str, Any]:
    results = pinqy(exploration_results.get('results', []))
    if not results.to.any(): return {'error': 'no results to analyze'}
    page_type_analysis = results.group.group_by(lambda r: r['classification']['type'])
    type_summary = pinqy(page_type_analysis.items()).to.dict(
        key_selector=lambda item: item[0],
        value_selector=lambda item: (
            lambda pages=pinqy(item[1]),
                   all_data=pinqy(item[1]).select_many(lambda p: p.get('extraction', {}).get('extracted_data', [])):
            {
                'page_count': pages.to.count(),
                'avg_confidence': pages.stats.average(lambda p: p['classification']['confidence']),
                'total_data_items': all_data.to.count(),
                'avg_items_per_page': pages.stats.average(
                    lambda p: len(p.get('extraction', {}).get('extracted_data', []))),
                'pages_with_links': pages.where(lambda p: p.get('child_links')).to.count(),
                'avg_links_per_page': pages.stats.average(lambda p: len(p.get('child_links', []))),
                'sample_titles': pages.select(lambda p: p.get('title', '')[:50]).take(3).to.list(),
                'discovered_fields': list(
                    all_data.to.first().keys()) if all_data.to.any() and all_data.to.first() else []
            }
        )()
    )
    return {
        'summary': {
            'total_pages': results.to.count(),
            'total_data_items': results.stats.sum(lambda r: len(r.get('extraction', {}).get('extracted_data', []))),
            'page_types_discovered': list(page_type_analysis.keys()),
            'most_common_page_type': max(page_type_analysis, key=lambda k: len(page_type_analysis[k])),
            'pages_with_data': results.where(lambda r: r.get('extraction', {}).get('extracted_data')).to.count()
        },
        'page_type_analysis': type_summary,
        'data_sample': results.select_many(lambda r: r.get('extraction', {}).get('extracted_data', [])).take(
            5).to.list()
    }


def export_results(exploration_results: Dict[str, Any], config: ExplorationConfig) -> str:
    domain = exploration_results.get('domain', 'unknown')
    filename = config.output_file or f"exploration_{domain}_{datetime.now():%Y%m%d_%H%M%S}.{config.output_format}"
    if config.output_format == 'json':
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(exploration_results, f, indent=2, ensure_ascii=False)
    else:
        all_data = pinqy(exploration_results.get('results', [])).select_many(
            lambda r: [{'source_url': r['url'], 'page_type': r['classification']['type'], **item} for item in
                       r.get('extraction', {}).get('extracted_data', [])]
        ).to.list()
        if not all_data: return f"no data to export for {config.output_format}"
        if config.output_format == 'csv':
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=all_data[0].keys())
                writer.writeheader()
                writer.writerows(all_data)
        elif config.output_format == 'sqlite':
            with sqlite3.connect(filename) as conn:
                conn.execute(
                    'CREATE TABLE pages (id INTEGER PRIMARY KEY, url TEXT UNIQUE, title TEXT, page_type TEXT, confidence REAL, processed_at TEXT)')
                conn.execute(
                    'CREATE TABLE extracted_data (id INTEGER PRIMARY KEY, page_id INTEGER, data_json TEXT, FOREIGN KEY (page_id) REFERENCES pages (id))')
                for result in exploration_results['results']:
                    cursor = conn.execute(
                        'INSERT OR IGNORE INTO pages (url, title, page_type, confidence, processed_at) VALUES (?, ?, ?, ?, ?)',
                        (result['url'], result.get('title', ''), result['classification']['type'],
                         result['classification']['confidence'], result['processed_at']))
                    page_id = cursor.lastrowid
                    if page_id:
                        for item in result.get('extraction', {}).get('extracted_data', []):
                            conn.execute('INSERT INTO extracted_data (page_id, data_json) VALUES (?, ?)',
                                         (page_id, json.dumps(item)))
    return filename


class FunctionalQueryBuilder:
    def __init__(self, exploration_results: Dict[str, Any]):
        self.results = pinqy(exploration_results.get('results', []))
        self.current_query = self.results
        self.query_history = []

    def filter_by_page_type(self, page_type: str):
        self.current_query = self.current_query.where(lambda r: page_type in r['classification']['type'])
        self.query_history.append(f"filter_by_page_type('{page_type}')")
        return self

    def select_data_items(self):
        self.current_query = self.current_query.select_many(
            lambda r: [{'source_url': r['url'], 'page_type': r['classification']['type'], **item} for item in
                       r.get('extraction', {}).get('extracted_data', [])])
        self.query_history.append("select_data_items()")
        return self

    def select_and_clean_numeric(self, field_name: str, new_field_name: str):
        def clean_numeric(val):
            if not isinstance(val, str): return None
            cleaned = re.sub(r'[^\d.]', '', val)
            try:
                return float(cleaned)
            except (ValueError, TypeError):
                return None

        self.current_query = self.current_query.select(
            lambda item: {**item, new_field_name: clean_numeric(item.get(field_name))}).where(
            lambda item: item[new_field_name] is not None)
        self.query_history.append(f"select_and_clean_numeric('{field_name}', '{new_field_name}')")
        return self

    def get_stats_for(self, field_name: str) -> Dict[str, Any]:
        query = self.current_query.where(lambda item: isinstance(item.get(field_name), (int, float)))
        if not query.to.any(): return {'count': 0}
        return {'count': query.to.count(), 'min': query.stats.min(lambda i: i[field_name]),
                'max': query.stats.max(lambda i: i[field_name]),
                'average': query.stats.average(lambda i: i[field_name]),
                'std_dev': query.stats.std_dev(lambda i: i[field_name]),
                'median': query.stats.median(lambda i: i[field_name])}

    def count(self) -> int:
        return self.current_query.to.count()

    def take(self, n: int) -> List[Any]:
        return self.current_query.take(n).to.list()

    def execute(self) -> List[Any]:
        return self.current_query.to.list()

    def reset(self):
        self.current_query = self.results; self.query_history = []; return self

    def get_query_string(self) -> str:
        return " -> ".join(self.query_history) or "base_query"

    def print_summary(self):
        print(f"\n{_c.info}query: {_c.warn}{self.get_query_string()}{_c.reset}")
        print(f"{_c.info}results: {_c.ok}{self.count()} items{_c.reset}")
        if self.count() > 0: print(f"{_c.grey}sample results:{_c.reset}"); [
            print(f"  {_c.grey}{i}. {str(item)[:150]}...{_c.reset}") for i, item in enumerate(self.take(3), 1)]


def create_cli_interface():
    import argparse
    parser = argparse.ArgumentParser(description='Functional Web Graph Explorer',
                                     epilog='Examples:\n  python web_exp.py https://pokemondb.net/pokedex/all --max-pages 50\n  python web_exp.py https://books.toscrape.com --output csv')
    parser.add_argument('start_url', help='Starting URL to explore')
    parser.add_argument('--max-depth', type=int, default=2, help='Max depth (default: 2)')
    parser.add_argument('--max-pages', type=int, default=100, help='Max pages (default: 100)')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between requests (default: 1.0)')
    parser.add_argument('--concurrent', type=int, default=3, help='Max concurrent requests (default: 3)')
    parser.add_argument('--output', choices=['json', 'csv', 'sqlite'], default='json', help='Output format')
    parser.add_argument('--output-file', help='Output filename')
    parser.add_argument('--no-robots', action='store_true', help='Ignore robots.txt')
    parser.add_argument('--user-agent', default='FunctionalWebExplorer/1.0', help='Custom user agent')
    parser.add_argument('--interactive', action='store_true', help='Start interactive query mode after exploration')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    return parser


def interactive_query_session(exploration_results: Dict[str, Any]):
    print(f"\n{_c.info}=== Interactive Query Mode ==={_c.reset}")
    print(f"  {_c.ok}.type <type>{_c.reset}        - filter by page type (e.g., .type product_list)")
    print(f"  {_c.ok}.select_data{_c.reset}        - transform query to data items")
    print(f"  {_c.ok}.clean <f> <new_f>{_c.reset} - clean field to numeric (e.g., .clean price price_f)")
    print(f"  {_c.ok}.stats <field>{_c.reset}      - get statistics for a numeric field (e.g., .stats price_f)")
    print(
        f"  {_c.ok}.count{_c.reset}, {_c.ok}.take <n>{_c.reset}, {_c.ok}.exec{_c.reset}, {_c.ok}.reset{_c.reset}, {_c.ok}.quit{_c.reset}")
    builder = FunctionalQueryBuilder(exploration_results)
    builder.print_summary()
    while True:
        try:
            command_str = input(f"\n{_c.warn}query>{_c.reset} ").strip()
            if not command_str: continue
            parts = command_str.split();
            command = parts[0];
            args = parts[1:]
            actions = {'.quit': lambda: 'break', '.reset': builder.reset,
                       '.count': lambda: print(f"Count: {builder.count()}"),
                       '.exec': lambda: pinqy(builder.execute()).util.for_each(
                           lambda x: print(json.dumps(x, indent=2))), '.summary': builder.print_summary,
                       '.select_data': builder.select_data_items,
                       '.type': lambda: builder.filter_by_page_type(args[0]) if args else print(
                           "Usage: .type <page_type>"),
                       '.take': lambda: pinqy(builder.take(int(args[0]))).util.for_each(
                           lambda x: print(json.dumps(x, indent=2))) if args else print("Usage: .take <n>"),
                       '.clean': lambda: builder.select_and_clean_numeric(args[0], args[1]) if len(
                           args) == 2 else print("Usage: .clean <field> <new_field>"),
                       '.stats': lambda: print(json.dumps(builder.get_stats_for(args[0]), indent=2)) if args else print(
                           "Usage: .stats <field>")}
            result = actions.get(command, lambda: print("Unknown command. Type .help for options."))()
            if result == 'break': break
            if command not in ['.count', '.exec', '.stats', '.take', '.quit']: builder.print_summary()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting..."); break
        except Exception as e:
            print(f"{_c.fail}Error: {e}{_c.reset}")


def main():
    parser = create_cli_interface()
    args = parser.parse_args()
    if args.verbose: logging.getLogger().setLevel(logging.DEBUG)
    config = ExplorationConfig(max_depth=args.max_depth, max_pages_per_domain=args.max_pages,
                               delay_between_requests=args.delay, max_concurrent_requests=args.concurrent,
                               respect_robots_txt=not args.no_robots, user_agent=args.user_agent,
                               output_format=args.output, output_file=args.output_file)
    print(f"{_c.info}functional web graph explorer{_c.reset}")
    print(f"exploring: {_c.warn}{args.start_url}{_c.reset}")
    start_time = time.time()
    exploration_results = explore_domain_functionally([args.start_url], config)
    end_time = time.time()
    if 'error' in exploration_results:
        print(f"{_c.fail}exploration failed: {exploration_results['error']}{_c.reset}")
        return
    analysis = analyze_exploration_results(exploration_results)
    print(f"\n{_c.ok}=== exploration complete ({end_time - start_time:.1f}s) ==={_c.reset}")
    print(f"domain: {_c.info}{exploration_results.get('domain', 'n/a')}{_c.reset}")
    print(f"pages processed: {_c.warn}{exploration_results.get('total_pages_processed', 0)}{_c.reset}")
    if 'error' not in analysis and 'summary' in analysis:
        summary = analysis['summary']
        print(f"data items extracted: {_c.ok}{summary['total_data_items']}{_c.reset}")
        print(f"page types discovered: {_c.info}{', '.join(summary.get('page_types_discovered', []))}{_c.reset}")
        print(f"\n{_c.info}page type analysis:{_c.reset}")
        for page_type, stats in analysis.get('page_type_analysis', {}).items():
            print(
                f"  {_c.ok}{page_type:<20}{_c.reset}: {stats.get('page_count', 0)} pages, " f"{stats.get('total_data_items', 0)} items, " f"confidence {stats.get('avg_confidence', 0):.2f}")
    output_file = export_results(exploration_results, config)
    print(f"\n{_c.ok}results exported to: {_c.warn}{output_file}{_c.reset}")
    if args.interactive:
        interactive_query_session(exploration_results)


if __name__ == "__main__":
    main()