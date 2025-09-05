#!/usr/bin/env python3
"""
functional web graph explorer tool
pure functional approach to web scraping at scale
everything is a tree in a graph of trees, discovered through structural analysis

key insight: domains are navigable objects that can be recursively explored
without presumptions about structure or data. tag-agnostic, dictionary-based,
purely functional approach using pinqy for linq-like operations.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs
from urllib.robotparser import RobotFileParser
from typing import Dict, List, Any, Callable, Optional, Tuple, Set
import json
import time
import re
from dataclasses import dataclass, asdict
from datetime import datetime
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
import hashlib
import sqlite3
from pinqy import from_iterable as pinqy, from_range

# configure minimal logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ExplorationConfig:
    """configuration for functional exploration"""
    max_depth: int = 3
    max_pages_per_domain: int = 100
    delay_between_requests: float = 1.0
    respect_robots_txt: bool = True
    max_concurrent_requests: int = 3
    user_agent: str = 'FunctionalWebExplorer/1.0'
    output_format: str = 'json'  # json, csv, sqlite
    output_file: Optional[str] = None
    custom_headers: Dict[str, str] = None

    def __post_init__(self):
        if self.custom_headers is None:
            self.custom_headers = {}


# functional domain model - everything is a dictionary
def create_session(config: ExplorationConfig) -> requests.Session:
    """pure function to create configured session"""
    session = requests.Session()
    session.headers.update({
        'User-Agent': config.user_agent,
        **config.custom_headers
    })
    return session


def check_robots_txt(domain: str, session: requests.Session) -> Callable[[str], bool]:
    """return a function that checks if url is allowed by robots.txt"""
    try:
        rp = RobotFileParser()
        rp.set_url(f"https://{domain}/robots.txt")
        rp.read()
        return lambda url: rp.can_fetch(session.headers.get('User-Agent', '*'), url)
    except:
        return lambda url: True  # allow all if can't fetch robots.txt


def fetch_page_safe(url: str, session: requests.Session, robots_checker: Callable[[str], bool]) -> Optional[
    BeautifulSoup]:
    """atomic: safely fetch and parse a single page"""
    if not robots_checker(url):
        logger.info(f"robots.txt blocks: {url}")
        return None

    try:
        response = session.get(url, timeout=15, allow_redirects=True)
        response.raise_for_status()

        # check content type
        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' not in content_type:
            return None

        return BeautifulSoup(response.content, 'lxml')
    except Exception as e:
        logger.warning(f"failed to fetch {url}: {str(e)[:100]}")
        return None


def element_to_functional_dict(element) -> Dict[str, Any]:
    """convert any element to a queryable functional dictionary"""
    if not hasattr(element, 'name'):
        text_content = str(element).strip()
        return {
            'type': 'text',
            'content': text_content,
            'length': len(text_content),
            'has_numbers': bool(re.search(r'\d', text_content)),
            'has_currency': bool(re.search(r'[\$\€\£\¥]', text_content)),
            'word_count': len(text_content.split()) if text_content else 0
        } if text_content else None

    # Only process actual HTML elements (not NavigableString objects)
    if not hasattr(element, 'attrs'):
        return None

    attrs = dict(element.attrs) if element.attrs else {}
    text = element.get_text(strip=True) if hasattr(element, 'get_text') else ''
    children = [element_to_functional_dict(child) for child in element.children if str(child).strip()]
    children = [c for c in children if c is not None]  # filter none

    return {
        'type': 'element',
        'tag': element.name,
        'attrs': attrs,
        'classes': attrs.get('class', []),
        'id': attrs.get('id'),
        'text': text,
        'text_length': len(text),
        'children': children,
        'children_count': len(children),
        'has_text': bool(text),
        'has_children': bool(children),
        'has_links': element.name == 'a' and 'href' in attrs,
        'href': attrs.get('href') if element.name == 'a' else None,
        'depth': 0  # will be set during tree traversal
    }


def set_element_depths(tree_dict: Dict[str, Any], depth: int = 0) -> Dict[str, Any]:
    """recursively set depth for each element in tree"""
    if tree_dict is None:
        return None

    result = tree_dict.copy()
    result['depth'] = depth

    if result.get('children'):
        result['children'] = [
            set_element_depths(child, depth + 1)
            for child in result['children']
            if child is not None
        ]

    return result


def page_to_functional_tree(soup: BeautifulSoup, base_url: str) -> Dict[str, Any]:
    """convert page to complete functional tree with metadata"""
    body = soup.body if soup.body else soup
    tree = element_to_functional_dict(body)

    if tree is None:
        return None

    # set depths and add page metadata
    tree = set_element_depths(tree)
    tree['base_url'] = base_url
    tree['title'] = soup.title.get_text(strip=True) if soup.title else ''
    tree['meta_description'] = ''

    meta_desc = soup.find('meta', attrs={'name': 'description'})
    if meta_desc:
        tree['meta_description'] = meta_desc.get('content', '')

    return tree


# pattern discovery
def extract_all_elements(tree_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """flatten tree to list of all elements"""
    if not tree_dict or tree_dict.get('type') != 'element':
        return []

    def collect_recursive(node: Dict[str, Any]) -> List[Dict[str, Any]]:
        elements = [node]
        children = node.get('children', [])
        for child in children:
            if child and child.get('type') == 'element':
                elements.extend(collect_recursive(child))
        return elements

    return collect_recursive(tree_dict)


def create_structural_signature(elem: Dict[str, Any]) -> str:
    """create unique structural fingerprint for element"""
    if elem.get('type') != 'element':
        return 'text'

    tag = elem.get('tag', '')
    classes = sorted(elem.get('classes', []))
    class_sig = '.'.join(classes[:3])  # limit to first 3 classes for manageability

    # child signature - what types of children does this have?
    children = elem.get('children', [])
    child_tags = pinqy(children).where(
        lambda c: c and c.get('type') == 'element'
    ).select(
        lambda c: c.get('tag', '')
    ).distinct().order_by(lambda x: x).to_list()

    child_sig = ','.join(child_tags)

    # structural features
    has_text = 'T' if elem.get('has_text') else 'N'
    has_links = 'L' if elem.get('has_links') else 'N'
    children_count = min(elem.get('children_count', 0), 10)  # cap for signature

    return f"{tag}|{class_sig}|{child_sig}|{has_text}{has_links}{children_count}"


def discover_repeating_patterns(tree_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """find structural patterns that repeat"""
    try:
        all_elements = extract_all_elements(tree_dict)
        if not all_elements:
            return []

        # group by structural signature using pinqy
        signature_groups = pinqy(all_elements).group_by(create_structural_signature)

        patterns = []

        for signature, group_elements in signature_groups.items():
            if len(group_elements) < 3:  # must repeat at least 3 times
                continue

            # calc statistics safely
            try:
                depths = [e.get('depth', 0) for e in group_elements]
                avg_depth = sum(depths) / len(depths) if depths else 0

                depths_pinqy = pinqy(depths)
                std_dev_val = depths_pinqy.std_dev() if len(depths) > 1 else 0

                has_consistent_text = all(bool(e.get('text')) for e in group_elements)
                consistent_links = sum(1 for e in group_elements if e.get('has_links'))

                pattern_quality_score = len(group_elements) * (1 + (1 / (1 + std_dev_val)))

                pattern = {
                    'signature': signature,
                    'elements': group_elements,
                    'count': len(group_elements),
                    'sample_element': group_elements[0],
                    'avg_depth': avg_depth,
                    'depth_variance': std_dev_val,
                    'has_consistent_text': has_consistent_text,
                    'consistent_links': consistent_links,
                    'pattern_quality_score': pattern_quality_score
                }
                patterns.append(pattern)

            except Exception as e:
                logger.error(f"Error processing group with signature '{signature[:50]}...': {e}", exc_info=True)
                continue

        # sort by quality score
        patterns.sort(key=lambda p: p['pattern_quality_score'], reverse=True)
        return patterns

    except Exception as e:
        logger.error(f"Exception in discover_repeating_patterns: {e}", exc_info=True)
        return []


def classify_page_functionally(tree_dict: Dict[str, Any]) -> Dict[str, Any]:
    """classify page type through data alone"""
    if not tree_dict:
        return {'type': 'invalid', 'confidence': 0.0, 'patterns': []}

    try:
        patterns = discover_repeating_patterns(tree_dict)

        if not patterns:
            return {
                'type': 'simple',
                'confidence': 0.3,
                'patterns': [],
                'reasoning': 'no repeating patterns found'
            }

        dominant_pattern = patterns[0]

        print(f"DEBUG: dominant_pattern keys: {dominant_pattern.keys()}")
        print(f"DEBUG: dominant_pattern types: {[(k, type(v)) for k, v in dominant_pattern.items()]}")

        # classification logic - be very explicit about types
        count = int(dominant_pattern['count'])
        has_links = bool(dominant_pattern['consistent_links'] > 0)
        depth_consistency = bool(dominant_pattern['depth_variance'] < 1.0)

        print(f"DEBUG: count={count} (type: {type(count)})")
        print(f"DEBUG: has_links={has_links} (type: {type(has_links)})")
        print(f"DEBUG: depth_consistency={depth_consistency} (type: {type(depth_consistency)})")

        # classification rules
        def rule1():
            return count >= 10 and has_links and depth_consistency

        def rule2():
            return count >= 5 and has_links

        def rule3():
            return count >= 8 and not has_links

        def rule4():
            return count >= 3 and depth_consistency

        def rule5():
            return True  # fallback

        classification_rules = [
            (rule1, 'list', 0.9),
            (rule2, 'directory', 0.8),
            (rule3, 'content_list', 0.7),
            (rule4, 'structured', 0.6),
            (rule5, 'detail', 0.4)
        ]

        # find the first matching rule
        page_type = 'unknown'
        confidence = 0.2

        for rule_func, p_type, conf in classification_rules:
            if rule_func():
                page_type = p_type
                confidence = conf
                break

        return {
            'type': page_type,
            'confidence': confidence,
            'patterns': patterns,
            'dominant_pattern': dominant_pattern,
            'reasoning': f"found {count} repeating elements with {dominant_pattern['consistent_links']} links"
        }

    except Exception as e:
        print(f"DEBUG: Exception in classify_page_functionally: {e}")
        print(f"DEBUG: Exception type: {type(e)}")
        import traceback
        traceback.print_exc()
        return {'type': 'error', 'confidence': 0.0, 'patterns': [], 'error': str(e)}


def discover_data_fields_functional(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """discover extractable data fields using functional composition"""
    if not elements:
        return []

    def extract_text_paths(elem: Dict[str, Any], path: str = "") -> List[Dict[str, Any]]:
        """recursively extract all text-bearing paths"""
        paths = []

        if elem.get('type') == 'text' and elem.get('content'):
            return [{'path': path, 'content': elem['content'], 'element': elem}]

        if elem.get('type') == 'element':
            tag = elem.get('tag', '')
            classes = '.'.join(elem.get('classes', [])[:2])  # limit classes
            current_path = f"{path}.{tag}"
            if classes:
                current_path += f".{classes}"

            # if element has direct text and no children, it's a leaf
            if elem.get('text') and not elem.get('children'):
                paths.append({
                    'path': current_path,
                    'content': elem['text'],
                    'element': elem,
                    'has_currency': elem.get('text', '').count('$') > 0 or elem.get('text', '').count('€') > 0,
                    'has_numbers': bool(re.search(r'\d', elem.get('text', ''))),
                    'word_count': len(elem.get('text', '').split())
                })

            # recurse through children
            for child in elem.get('children', []):
                if child:
                    paths.extend(extract_text_paths(child, current_path))

        return paths

    # extract paths from all sample elements
    all_paths = []
    sample_elements = elements[:min(10, len(elements))]  # sample for performance

    for elem in sample_elements:
        all_paths.extend(extract_text_paths(elem))

    if not all_paths:
        return []

    # group by path and find consistent fields
    path_groups = pinqy(all_paths).group_by(lambda p: p['path'])

    consistent_fields = (pinqy(path_groups.items())
                         .where(lambda group: len(group[1]) >= len(sample_elements) * 0.5)  # appear in majority
                         .select(lambda group: {
        'field_path': group[0],
        'occurrence_rate': len(group[1]) / len(sample_elements),
        'sample_values': [p['content'][:100] for p in group[1][:3]],  # truncate samples
        'avg_length': pinqy(group[1]).average(lambda p: len(p['content'])),
        'has_currency': pinqy(group[1]).any(lambda p: p.get('has_currency', False)),
        'has_numbers': pinqy(group[1]).any(lambda p: p.get('has_numbers', False)),
        'avg_word_count': pinqy(group[1]).average(lambda p: p.get('word_count', 0)),
        'field_type': 'currency' if pinqy(group[1]).any(lambda p: p.get('has_currency', False)) else
        'numeric' if pinqy(group[1]).any(lambda p: p.get('has_numbers', False)) else
        'title' if pinqy(group[1]).average(lambda p: p.get('word_count', 0)) < 10 else
        'description'
    })
                         .order_by_descending(lambda f: f['occurrence_rate'])
                         .to_list())

    return consistent_fields


def discover_link_patterns_functional(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """discover link patterns"""

    def extract_links(elem: Dict[str, Any], path: str = "") -> List[Dict[str, Any]]:
        links = []

        if elem.get('type') == 'element':
            tag = elem.get('tag', '')
            current_path = f"{path}.{tag}"

            if elem.get('has_links') and elem.get('href'):
                links.append({
                    'path': current_path,
                    'href': elem['href'],
                    'text': elem.get('text', ''),
                    'element': elem
                })

            # recurse through children
            for child in elem.get('children', []):
                if child:
                    links.extend(extract_links(child, current_path))

        return links

    all_links = []
    sample_elements = elements[:min(10, len(elements))]

    for elem in sample_elements:
        all_links.extend(extract_links(elem))

    if not all_links:
        return []

    # group by path pattern
    path_groups = pinqy(all_links).group_by(lambda l: l['path'])

    consistent_links = (pinqy(path_groups.items())
                        .where(lambda group: len(group[1]) >= len(sample_elements) * 0.4)
                        .select(lambda group: {
        'link_path': group[0],
        'occurrence_rate': len(group[1]) / len(sample_elements),
        'sample_hrefs': [l['href'] for l in group[1][:3]],
        'sample_texts': [l['text'][:50] for l in group[1][:3]],
        'href_patterns': pinqy(group[1]).select(lambda l: l['href']).distinct().count(),
        'is_relative': pinqy(group[1]).all(lambda l: not l['href'].startswith('http')),
        'avg_text_length': pinqy(group[1]).average(lambda l: len(l['text']))
    })
                        .order_by_descending(lambda l: l['occurrence_rate'])
                        .to_list())

    return consistent_links


def create_functional_extractor(classification: Dict[str, Any]) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """create data extraction function based on discovered patterns"""
    patterns = classification.get('patterns', [])
    page_type = classification.get('type', 'unknown')

    if not patterns:
        return lambda tree: {'type': page_type, 'extracted_data': [], 'links': []}

    dominant_pattern = patterns[0]

    def extract_from_tree(tree_dict: Dict[str, Any]) -> Dict[str, Any]:
        # find elements matching the dominant pattern
        all_elements = extract_all_elements(tree_dict)
        signature = dominant_pattern['signature']

        matching_elements = pinqy(all_elements).where(
            lambda e: create_structural_signature(e) == signature
        ).to_list()

        if not matching_elements:
            return {'type': page_type, 'extracted_data': [], 'links': []}

        # discover fields and links dynamically
        data_fields = discover_data_fields_functional(matching_elements)
        link_fields = discover_link_patterns_functional(matching_elements)

        # extract data from each matching element
        extracted_items = []
        extracted_links = set()

        for elem in matching_elements:
            item_data = {'_signature': signature, '_element_depth': elem.get('depth', 0)}

            # extract data fields
            for field in data_fields:
                # TODO: this is simplified - i need to implement the logic to traverse the path
                # for now, extract based on the pattern
                field_name = field.get('field_type', 'field')
                if field['sample_values']:
                    item_data[field_name] = field['sample_values'][0]  # simplified

            # extract links
            for link_field in link_fields:
                if link_field['sample_hrefs']:
                    href = link_field['sample_hrefs'][0]  # simplified
                    if href:
                        item_data['link'] = href
                        extracted_links.add(href)

            extracted_items.append(item_data)

        return {
            'type': page_type,
            'confidence': classification.get('confidence', 0.0),
            'extracted_data': extracted_items,
            'links': list(extracted_links),
            'data_fields_discovered': len(data_fields),
            'link_patterns_discovered': len(link_fields),
            'total_items_extracted': len(extracted_items)
        }

    return extract_from_tree


def process_url_functionally(url: str, session: requests.Session, robots_checker: Callable[[str], bool],
                             base_domain: str) -> Optional[Dict[str, Any]]:
    """process a single url through the complete pipeline"""
    soup = fetch_page_safe(url, session, robots_checker)
    if not soup:
        return None

    # convert to functional tree
    tree = page_to_functional_tree(soup, url)
    if not tree:
        return None

    # classify the page type
    classification = classify_page_functionally(tree)

    # create and apply extractor
    extractor = create_functional_extractor(classification)
    extraction_result = extractor(tree)

    # resolve relative links to absolute
    links = extraction_result.get('links', [])
    absolute_links = []

    for link in links:
        if link:
            absolute_url = urljoin(url, link)
            parsed = urlparse(absolute_url)
            if parsed.netloc == base_domain or not parsed.netloc:
                absolute_links.append(absolute_url)

    return {
        'url': url,
        'title': tree.get('title', ''),
        'meta_description': tree.get('meta_description', ''),
        'classification': classification,
        'extraction': extraction_result,
        'child_links': absolute_links,
        'processed_at': datetime.now().isoformat()
    }


def explore_domain_functionally(start_urls: List[str], config: ExplorationConfig) -> Dict[str, Any]:
    """main functional exploration pipeline"""
    if not start_urls:
        return {'error': 'no start urls provided'}

    # extract domain from first url
    parsed = urlparse(start_urls[0])
    domain = parsed.netloc

    logger.info(f"exploring domain: {domain}")
    logger.info(f"config: {asdict(config)}")

    # setup
    session = create_session(config)
    robots_checker = check_robots_txt(domain, session) if config.respect_robots_txt else lambda x: True

    # tracking state
    all_results = []
    visited_urls = set()
    url_queue = start_urls.copy()
    pages_processed = 0

    def process_batch(urls: List[str]) -> List[Dict[str, Any]]:
        """process batch of urls concurrently"""
        results = []

        with ThreadPoolExecutor(max_workers=config.max_concurrent_requests) as executor:
            future_to_url = {
                executor.submit(process_url_functionally, url, session, robots_checker, domain): url
                for url in urls
            }

            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result(timeout=30)
                    if result:
                        results.append(result)
                        logger.info(f"processed: {url} -> {result['classification']['type']}")
                    else:
                        logger.warning(f"failed to process: {url}")
                except Exception as e:
                    logger.error(f"error processing {url}: {str(e)[:100]}")

        return results

    # main exploration loop
    current_depth = 0

    while url_queue and current_depth < config.max_depth and pages_processed < config.max_pages_per_domain:
        # get next batch of urls
        batch_size = min(config.max_concurrent_requests, len(url_queue),
                         config.max_pages_per_domain - pages_processed)

        current_batch = []
        for _ in range(batch_size):
            if url_queue:
                url = url_queue.pop(0)
                if url not in visited_urls:
                    current_batch.append(url)
                    visited_urls.add(url)

        if not current_batch:
            break

        logger.info(f"processing batch of {len(current_batch)} urls at depth {current_depth}")

        # process batch
        batch_results = process_batch(current_batch)
        all_results.extend(batch_results)
        pages_processed += len(batch_results)

        # collect new links for next depth
        if current_depth < config.max_depth - 1:
            new_links = pinqy(batch_results).select_many(
                lambda r: r.get('child_links', [])
            ).where(
                lambda link: link not in visited_urls
            ).distinct().to_list()

            url_queue.extend(new_links)
            logger.info(f"discovered {len(new_links)} new links")

        current_depth += 1

        # respect rate limiting
        if config.delay_between_requests > 0 and current_batch:
            time.sleep(config.delay_between_requests)

    return {
        'domain': domain,
        'total_pages_processed': pages_processed,
        'total_urls_visited': len(visited_urls),
        'exploration_depth': current_depth,
        'results': all_results,
        'config_used': asdict(config),
        'completed_at': datetime.now().isoformat()
    }


def analyze_exploration_results(exploration_results: Dict[str, Any]) -> Dict[str, Any]:
    """analyze exploration results using functional composition"""
    results = pinqy(exploration_results.get('results', []))

    if not results.any():
        return {'error': 'no results to analyze'}

    # analyze page types
    page_type_analysis = results.group_by(lambda r: r['classification']['type'])

    type_summary = {}
    for page_type, pages in page_type_analysis.items():
        pages_pinqy = pinqy(pages)

        # extract all data items
        all_data = pages_pinqy.select_many(lambda p: p.get('extraction', {}).get('extracted_data', []))

        type_summary[page_type] = {
            'page_count': len(pages),
            'avg_confidence': pages_pinqy.average(lambda p: p['classification']['confidence']),
            'total_data_items': all_data.count(),
            'avg_items_per_page': pages_pinqy.average(lambda p: len(p.get('extraction', {}).get('extracted_data', []))),
            'pages_with_links': pages_pinqy.where(lambda p: p.get('child_links')).count(),
            'avg_links_per_page': pages_pinqy.average(lambda p: len(p.get('child_links', []))),
            'sample_titles': pages_pinqy.select(lambda p: p.get('title', '')[:50]).take(3).to_list()
        }

        # analyze discovered fields if any data exists
        if all_data.any():
            sample_data = all_data.first()
            type_summary[page_type]['discovered_fields'] = list(sample_data.keys()) if sample_data else []

    # overall statistics
    total_data_items = results.sum(lambda r: len(r.get('extraction', {}).get('extracted_data', [])))

    return {
        'summary': {
            'total_pages': results.count(),
            'total_data_items': total_data_items,
            'page_types_discovered': list(page_type_analysis.keys()),
            'most_common_page_type': max(page_type_analysis.keys(), key=lambda k: len(page_type_analysis[k])),
            'pages_with_data': results.where(
                lambda r: len(r.get('extraction', {}).get('extracted_data', [])) > 0).count()
        },
        'page_type_analysis': type_summary,
        'data_sample': results.select_many(lambda r: r.get('extraction', {}).get('extracted_data', [])).take(
            5).to_list()
    }


def export_results(exploration_results: Dict[str, Any], config: ExplorationConfig) -> str:
    """export results in specified format"""
    results = exploration_results.get('results', [])
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    domain = exploration_results.get('domain', 'unknown')

    if config.output_format == 'json':
        filename = config.output_file or f"exploration_{domain}_{timestamp}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(exploration_results, f, indent=2, ensure_ascii=False)
        return filename

    elif config.output_format == 'csv':
        filename = config.output_file or f"exploration_{domain}_{timestamp}.csv"

        # flatten all extracted data
        all_data = pinqy(results).select_many(
            lambda r: [
                {
                    'source_url': r['url'],
                    'page_type': r['classification']['type'],
                    'page_confidence': r['classification']['confidence'],
                    **item
                } for item in r.get('extraction', {}).get('extracted_data', [])
            ]
        ).to_list()

        if all_data:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=all_data[0].keys())
                writer.writeheader()
                writer.writerows(all_data)

        return filename

    elif config.output_format == 'sqlite':
        filename = config.output_file or f"exploration_{domain}_{timestamp}.db"

        conn = sqlite3.connect(filename)

        # create tables -- yeah, i'm a *freak* for this.
        # db off-loading baby ૮ ˶ᵔ ᵕ ᵔ˶ ა
        conn.execute('''
                     CREATE TABLE pages
                     (
                         id           INTEGER PRIMARY KEY,
                         url          TEXT UNIQUE,
                         title        TEXT,
                         page_type    TEXT,
                         confidence   REAL,
                         processed_at TEXT
                     )
                     ''')

        conn.execute('''
                     CREATE TABLE extracted_data
                     (
                         id        INTEGER PRIMARY KEY,
                         page_id   INTEGER,
                         data_json TEXT,
                         FOREIGN KEY (page_id) REFERENCES pages (id)
                     )
                     ''')

        # insert data
        for result in results:
            cursor = conn.execute(
                'INSERT OR IGNORE INTO pages (url, title, page_type, confidence, processed_at) VALUES (?, ?, ?, ?, ?)',
                (result['url'], result.get('title', ''), result['classification']['type'],
                 result['classification']['confidence'], result['processed_at'])
            )

            page_id = cursor.lastrowid

            # insert extracted data
            for item in result.get('extraction', {}).get('extracted_data', []):
                conn.execute(
                    'INSERT INTO extracted_data (page_id, data_json) VALUES (?, ?)',
                    (page_id, json.dumps(item))
                )

        conn.commit()
        conn.close()

        return filename

    return "unknown format"


# interactive query builder
class FunctionalQueryBuilder:
    """interactive query builder"""

    def __init__(self, exploration_results: Dict[str, Any]):
        self.results = pinqy(exploration_results.get('results', []))
        self.current_query = self.results
        self.query_history = []

    def filter_by_page_type(self, page_type: str) -> 'FunctionalQueryBuilder':
        """filter results by page type"""
        self.current_query = self.current_query.where(
            lambda r: r['classification']['type'] == page_type
        )
        self.query_history.append(f"filter_by_page_type('{page_type}')")
        return self

    def filter_by_confidence(self, min_confidence: float) -> 'FunctionalQueryBuilder':
        """filter by minimum confidence threshold"""
        self.current_query = self.current_query.where(
            lambda r: r['classification']['confidence'] >= min_confidence
        )
        self.query_history.append(f"filter_by_confidence({min_confidence})")
        return self

    def with_data_items(self) -> 'FunctionalQueryBuilder':
        """filter to pages that have extracted data"""
        self.current_query = self.current_query.where(
            lambda r: len(r.get('extraction', {}).get('extracted_data', [])) > 0
        )
        self.query_history.append("with_data_items()")
        return self

    def with_links(self) -> 'FunctionalQueryBuilder':
        """filter to pages that have child links"""
        self.current_query = self.current_query.where(
            lambda r: len(r.get('child_links', [])) > 0
        )
        self.query_history.append("with_links()")
        return self

    def select_data_items(self) -> 'FunctionalQueryBuilder':
        """transform to extracted data items"""
        self.current_query = self.current_query.select_many(
            lambda r: [
                {
                    'source_url': r['url'],
                    'page_type': r['classification']['type'],
                    **item
                }
                for item in r.get('extraction', {}).get('extracted_data', [])
            ]
        )
        self.query_history.append("select_data_items()")
        return self

    def group_by_field(self, field_name: str) -> Dict[str, List[Any]]:
        """group current results by specified field"""
        result = self.current_query.group_by(lambda item: item.get(field_name, 'unknown'))
        self.query_history.append(f"group_by_field('{field_name}')")
        return result

    def count(self) -> int:
        """get count of current query results"""
        return self.current_query.count()

    def take(self, n: int) -> List[Any]:
        """take first n results"""
        self.query_history.append(f"take({n})")
        return self.current_query.take(n).to_list()

    def execute(self) -> List[Any]:
        """execute current query and return results"""
        return self.current_query.to_list()

    def reset(self) -> 'FunctionalQueryBuilder':
        """reset query to original results"""
        self.current_query = self.results
        self.query_history = []
        return self

    def get_query_string(self) -> str:
        """get string representation of current query chain"""
        return " -> ".join(self.query_history) if self.query_history else "base_query"

    def print_summary(self) -> None:
        """print summary of current query results"""
        count = self.count()
        query_str = self.get_query_string()

        print(f"\nquery: {query_str}")
        print(f"results: {count} items")

        if count > 0:
            sample = self.take(3)
            print(f"sample results:")
            for i, item in enumerate(sample, 1):
                if isinstance(item, dict):
                    keys = list(item.keys())[:5]  # show first 5 keys
                    print(f"  {i}. {keys} ...")
                else:
                    print(f"  {i}. {str(item)[:100]} ...")


# command line interface for the tool
def create_cli_interface():
    """create command line interface for the functional web explorer"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Functional Web Graph Explorer - discover and extract data from websites',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python web_explorer.py https://example.com/products --max-depth 2 --output json
  python web_explorer.py https://news.site.com --max-pages 50 --delay 2.0
  python web_explorer.py https://catalog.com --concurrent 5 --output csv --output-file results.csv
        '''
    )

    parser.add_argument('start_url', help='Starting URL to explore')
    parser.add_argument('--max-depth', type=int, default=2, help='Maximum depth to explore (default: 2)')
    parser.add_argument('--max-pages', type=int, default=100, help='Maximum pages per domain (default: 100)')
    parser.add_argument('--delay', type=float, default=3.0, help='Delay between requests in seconds (default: 1.0)')
    parser.add_argument('--concurrent', type=int, default=3, help='Max concurrent requests (default: 3)')
    parser.add_argument('--output', choices=['json', 'csv', 'sqlite'], default='json',
                        help='Output format (default: json)')
    parser.add_argument('--output-file', help='Output filename (auto-generated if not specified)')
    parser.add_argument('--no-robots', action='store_true', help='Ignore robots.txt')
    parser.add_argument('--user-agent', default='FunctionalWebExplorer/1.0', help='Custom user agent')
    parser.add_argument('--interactive', action='store_true', help='Start interactive query mode after exploration')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')

    return parser


def interactive_query_session(exploration_results: Dict[str, Any]):
    """interactive query session for exploring results"""
    print("\n=== Interactive Query Mode ===")
    print("Available commands:")
    print("  .filter_type <type>     - Filter by page type")
    print("  .filter_confidence <n>  - Filter by minimum confidence")
    print("  .with_data             - Filter to pages with extracted data")
    print("  .with_links            - Filter to pages with links")
    print("  .select_data           - Transform to data items")
    print("  .count                 - Count current results")
    print("  .take <n>              - Show first n results")
    print("  .execute               - Execute query and show all results")
    print("  .reset                 - Reset to original results")
    print("  .summary               - Show query summary")
    print("  .help                  - Show this help")
    print("  .quit                  - Exit interactive mode")
    print()

    builder = FunctionalQueryBuilder(exploration_results)
    builder.print_summary()

    while True:
        try:
            command = input("\nquery> ").strip()

            if command == '.quit':
                break
            elif command == '.help':
                print("Available commands listed above")
            elif command == '.reset':
                builder.reset()
                print("Query reset to original results")
                builder.print_summary()
            elif command == '.count':
                print(f"Current result count: {builder.count()}")
            elif command == '.execute':
                results = builder.execute()
                print(f"Executed query returned {len(results)} results")
                if results:
                    for i, result in enumerate(results[:10], 1):  # show first 10
                        print(f"  {i}. {str(result)[:200]}...")
                    if len(results) > 10:
                        print(f"  ... and {len(results) - 10} more")
            elif command == '.summary':
                builder.print_summary()
            elif command == '.with_data':
                builder.with_data_items()
                builder.print_summary()
            elif command == '.with_links':
                builder.with_links()
                builder.print_summary()
            elif command == '.select_data':
                builder.select_data_items()
                builder.print_summary()
            elif command.startswith('.filter_type '):
                page_type = command.split(' ', 1)[1]
                builder.filter_by_page_type(page_type)
                builder.print_summary()
            elif command.startswith('.filter_confidence '):
                try:
                    confidence = float(command.split(' ', 1)[1])
                    builder.filter_by_confidence(confidence)
                    builder.print_summary()
                except ValueError:
                    print("Invalid confidence value. Please provide a number between 0 and 1.")
            elif command.startswith('.take '):
                try:
                    n = int(command.split(' ', 1)[1])
                    results = builder.take(n)
                    print(f"First {len(results)} results:")
                    for i, result in enumerate(results, 1):
                        print(f"  {i}. {str(result)[:200]}...")
                except ValueError:
                    print("Invalid number. Please provide an integer.")
            elif command:
                print(f"Unknown command: {command}. Type .help for available commands.")

        except KeyboardInterrupt:
            print("\nExiting interactive mode...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """main entry point for the functional web explorer tool"""
    parser = create_cli_interface()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # create configuration
    config = ExplorationConfig(
        max_depth=args.max_depth,
        max_pages_per_domain=args.max_pages,
        delay_between_requests=args.delay,
        max_concurrent_requests=args.concurrent,
        respect_robots_txt=not args.no_robots,
        user_agent=args.user_agent,
        output_format=args.output,
        output_file=args.output_file
    )

    print(f"functional web graph explorer")
    print(f"exploring: {args.start_url}")
    print(
        f"configuration: depth={config.max_depth}, pages={config.max_pages_per_domain}, delay={config.delay_between_requests}s")
    print()

    # run exploration
    start_time = time.time()
    exploration_results = explore_domain_functionally([args.start_url], config)
    end_time = time.time()

    if 'error' in exploration_results:
        print(f"exploration failed: {exploration_results['error']}")
        return

    # analyze results
    analysis = analyze_exploration_results(exploration_results)

    # print summary
    print(f"\n=== exploration complete ({end_time - start_time:.1f}s) ===")
    print(f"domain: {exploration_results['domain']}")
    print(f"pages processed: {exploration_results['total_pages_processed']}")
    print(f"urls visited: {exploration_results['total_urls_visited']}")
    print(f"exploration depth: {exploration_results['exploration_depth']}")

    if 'error' not in analysis:
        summary = analysis['summary']
        print(f"data items extracted: {summary['total_data_items']}")
        print(f"page types discovered: {', '.join(summary['page_types_discovered'])}")
        print(f"most common type: {summary['most_common_page_type']}")
        print(f"pages with data: {summary['pages_with_data']}")

        # show page type breakdown
        print(f"\npage type analysis:")
        for page_type, stats in analysis['page_type_analysis'].items():
            print(f"  {page_type}: {stats['page_count']} pages, "
                  f"{stats['total_data_items']} items, "
                  f"confidence {stats['avg_confidence']:.2f}")

    # export results
    output_file = export_results(exploration_results, config)
    print(f"\nresults exported to: {output_file}")

    # demonstrate functional queries on results
    print(f"\n=== functional query examples ===")

    # example queries using pinqy
    results_pinqy = pinqy(exploration_results['results'])

    # find all list-type pages
    list_pages = results_pinqy.where(
        lambda r: r['classification']['type'] == 'list'
    ).count()
    print(f"list-type pages: {list_pages}")

    # find pages with high confidence and data
    high_confidence_with_data = results_pinqy.where(
        lambda r: r['classification']['confidence'] > 0.7 and
                  len(r.get('extraction', {}).get('extracted_data', [])) > 0
    ).count()
    print(f"high confidence pages with data: {high_confidence_with_data}")

    # show top domains for child links
    all_child_links = results_pinqy.select_many(lambda r: r.get('child_links', []))
    if all_child_links.any():
        link_domains = all_child_links.select(
            lambda url: urlparse(url).netloc
        ).group_by(lambda domain: domain)

        top_domains = pinqy(link_domains.items()).order_by_descending(
            lambda item: len(item[1])
        ).take(3).select(
            lambda item: f"{item[0]} ({len(item[1])} links)"
        ).to_list()

        print(f"top child link domains: {', '.join(top_domains)}")

    # show sample of extracted data fields
    all_extracted_data = results_pinqy.select_many(
        lambda r: r.get('extraction', {}).get('extracted_data', [])
    )

    if all_extracted_data.any():
        sample_item = all_extracted_data.first()
        if sample_item:
            fields = list(sample_item.keys())
            print(f"sample extracted fields: {', '.join(fields)}")

    # interactive mode
    if args.interactive:
        interactive_query_session(exploration_results)


if __name__ == "__main__":
    main()


# additional utility functions for advanced use cases

def compare_explorations(exploration1: Dict[str, Any], exploration2: Dict[str, Any]) -> Dict[str, Any]:
    """functionally compare two exploration results"""
    results1 = pinqy(exploration1.get('results', []))
    results2 = pinqy(exploration2.get('results', []))

    # compare page type distributions
    types1 = results1.group_by(lambda r: r['classification']['type'])
    types2 = results2.group_by(lambda r: r['classification']['type'])

    all_types = set(types1.keys()) | set(types2.keys())

    type_comparison = {}
    for page_type in all_types:
        count1 = len(types1.get(page_type, []))
        count2 = len(types2.get(page_type, []))
        type_comparison[page_type] = {
            'exploration1': count1,
            'exploration2': count2,
            'difference': count2 - count1,
            'percent_change': ((count2 - count1) / count1 * 100) if count1 > 0 else 0
        }

    return {
        'domain1': exploration1.get('domain'),
        'domain2': exploration2.get('domain'),
        'type_comparison': type_comparison,
        'total_pages_diff': exploration2.get('total_pages_processed', 0) - exploration1.get('total_pages_processed', 0)
    }


def create_extraction_template(exploration_results: Dict[str, Any], page_type: str) -> Dict[str, Any]:
    """create reusable extraction template from discovered patterns"""
    results = pinqy(exploration_results.get('results', []))

    type_pages = results.where(
        lambda r: r['classification']['type'] == page_type
    ).to_list()

    if not type_pages:
        return {'error': f'no pages of type {page_type} found'}

    # analyze patterns across all pages of this type
    all_patterns = []
    for page in type_pages:
        patterns = page['classification'].get('patterns', [])
        if patterns:
            all_patterns.append(patterns[0])  # take dominant pattern

    if not all_patterns:
        return {'error': f'no patterns found for type {page_type}'}

    # find most common signature
    signature_groups = pinqy(all_patterns).group_by(lambda p: p['signature'])
    most_common = pinqy(signature_groups.items()).max(lambda item: len(item[1]))

    # create template
    template = {
        'page_type': page_type,
        'signature': most_common[0],
        'confidence': pinqy(most_common[1]).average(lambda p: p.get('pattern_quality_score', 0)),
        'sample_count': len(most_common[1]),
        'template_created_at': datetime.now().isoformat()
    }

    return template


def batch_explore_domains(domain_urls: List[str], config: ExplorationConfig) -> Dict[str, Any]:
    """explore multiple domains in batch using functional approach"""
    all_results = {}

    for domain_url in domain_urls:
        try:
            logger.info(f"exploring domain: {domain_url}")
            result = explore_domain_functionally([domain_url], config)
            domain = result.get('domain', urlparse(domain_url).netloc)
            all_results[domain] = result

            # respect rate limiting between domains
            if config.delay_between_requests > 0:
                time.sleep(config.delay_between_requests * 2)

        except Exception as e:
            logger.error(f"failed to explore {domain_url}: {e}")
            all_results[domain_url] = {'error': str(e)}

    return {
        'batch_results': all_results,
        'domains_processed': len(all_results),
        'successful_domains': pinqy(all_results.values()).where(lambda r: 'error' not in r).count(),
        'total_pages': pinqy(all_results.values()).where(lambda r: 'error' not in r).sum(
            lambda r: r.get('total_pages_processed', 0))
    }