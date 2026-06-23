import os
import xml.etree.ElementTree as etree
import copy

def extract_xml_subset(xml_input_file, xml_output_file, number_of_sentences=50):
    """Extract first N sentences from XML file and save to new XML file."""
    os.makedirs(os.path.dirname(xml_output_file), exist_ok=True)
    tree = etree.parse(xml_input_file)
    root = tree.getroot()
    
    # Define the TEI namespace
    namespace = {'tei': 'http://www.tei-c.org/ns/1.0', 'vici': 'http://www.tei-c.org/ns/VICI'}
    
    # Use XPath to find all <s> elements regardless of their position in the hierarchy
    all_sentences = root.findall('.//tei:s', namespace)
    
    # Extract only the first N sentences (deep copy to preserve them)
    selected_sentences = [copy.deepcopy(s) for s in all_sentences[:number_of_sentences]]
    
    # Get the namespace URI
    ns_uri = 'http://www.tei-c.org/ns/1.0'
    vici_ns = 'http://www.tei-c.org/ns/VICI'
    
    # Register namespaces to preserve prefixes
    etree.register_namespace('', ns_uri)
    etree.register_namespace('vici', vici_ns)
    
    # Create a new root element with proper namespace
    new_root = etree.Element(f'{{{ns_uri}}}TEI')
    
    # Copy the teiHeader from the original file
    tei_header = root.find('.//tei:teiHeader', namespace)
    if tei_header is not None:
        new_header = copy.deepcopy(tei_header)
        new_root.append(new_header)
    
    # Create the text structure
    text = etree.SubElement(new_root, f'{{{ns_uri}}}text')
    
    # Create group
    group = etree.SubElement(text, f'{{{ns_uri}}}group')
    
    # Create text element (nested under group)
    text_elem = etree.SubElement(group, f'{{{ns_uri}}}text')
    text_elem.set('xml:id', 'subset-fragment01')
    
    body = etree.SubElement(text_elem, f'{{{ns_uri}}}body')
    
    div1 = etree.SubElement(body, f'{{{ns_uri}}}div1')
    div1.set('n', 'subset extracted')
    div1.set('type', 'u')
    
    # Create p element to hold sentences
    p = etree.SubElement(div1, f'{{{ns_uri}}}p')
    
    # Add all selected sentences to the paragraph
    for s in selected_sentences:
        p.append(s)
    
    # Create a new tree with the modified root
    new_tree = etree.ElementTree(new_root)
    new_tree.write(xml_output_file, encoding='utf-8', xml_declaration=True)
    
    print(f"Extracted {len(selected_sentences)} sentences to {xml_output_file}")


def clean_word_text(text):
    """Clean word text by removing newlines and stripping whitespace."""
    if not text:
        return ''
    # Take content before the first newline, then strip leading/trailing whitespace.
    return text.split('\n')[0].strip()


def get_word_text(elem, seg_elem=None):
    """Extract word text from XML element, handling <seg> elements for metaphors."""
    if seg_elem is not None:
        prefix = clean_word_text(elem.text if elem.text else '')
        if not prefix.strip():
            prefix = ''
        # Strip trailing whitespace from seg_elem.text: the <seg> element stores the word form
        # and any trailing space inside it is a word-separator, not part of the word itself.
        # Spacing between words is handled separately by the add_trailing_space logic.
        seg_text = seg_elem.text.rstrip() if seg_elem.text else ''
        return prefix + seg_text
    return clean_word_text(elem.text if elem.text else '')


def is_classifiable_word(word_text):
    """Check if word contains alphanumeric characters."""
    return any(char.isalnum() for char in word_text)


def extract_vuamc_data(xml_file_path):
    """
    Extract sentences and word-level metaphor labels from VUAMC XML file.
    
    Returns three outputs:
    1. sentences: List of sentences as they appear in the text (with punctuation preserved)
    2. words_only: List of word lists (only words, no punctuation) 
    3. word_labels: List of label lists corresponding to words_only (1 for metaphor, 0 for non-metaphor)
    
    This separation allows:
    - Training on full texts with punctuation for context
    - Label alignment only for word head-tokens
    
    Args:
        xml_file_path (str): Path to the VUAMC XML file
        
    Returns:
        tuple: (sentences, words_only, word_labels)
               - sentences: List of sentence strings with punctuation
               - words_only: List of lists containing only words (no punctuation)
               - word_labels: List of label lists for words_only
    """
    tree = etree.parse(xml_file_path)
    root = tree.getroot()
    
    # Define namespace
    namespace = {'tei': 'http://www.tei-c.org/ns/1.0', 'vici': 'http://www.tei-c.org/ns/VICI'}
    
    # Find all sentence elements
    sentences_elements = root.findall('.//tei:s', namespace)
    
    sentences = []      # Full sentences with punctuation
    words_only = []     # Only words, no punctuation
    word_labels = []   # Labels corresponding to words_only
    
    for sent_elem in sentences_elements:
        sentence_parts = []  # All text parts (words + punctuation) as they appear
        current_words = []   # Only words (stripped)
        current_labels = []  # Labels for current_words
        
        # Get list of all children elements
        children = list(sent_elem)
        
        for i, elem in enumerate(children):
            # Resolve the local tag name (strip namespace prefix if present)
            elem_local = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag

            # ── Process word elements ────────────────────────────────────────
            if elem_local == 'w':
                word_type = elem.get('type', '')
                seg_elem = elem.find('.//seg[@function="mrw"][@type="met"]')
                if seg_elem is None:
                    seg_elem = elem.find('.//tei:seg[@function="mrw"][@type="met"]', namespace)

                if seg_elem is not None:
                    # Metaphor word: text lives inside <seg>
                    word_text = get_word_text(elem, seg_elem)
                    label = 1
                else:
                    # Literal word: skip whitespace-only placeholder elements
                    if elem.text and not elem.text.strip():
                        continue
                    word_text = get_word_text(elem)
                    label = 0

                # clean_word_text already strips, but be explicit for safety
                word_text = word_text.strip()
                if not word_text:
                    continue

                # ── Unified trailing-space logic (literal AND metaphor) ──────
                # Look ahead to the first *meaningful* element, skipping
                # whitespace-only placeholder <w> elements that contribute no
                # text (BNC artefacts). Space is added when the next token
                # starts an alphanumeric word or an opening bracket; it is
                # omitted before punctuation (which carries its own spacing).
                next_meaningful = None
                for j in range(i + 1, len(children)):
                    cand = children[j]
                    cand_tag = cand.tag.split('}')[-1] if '}' in cand.tag else cand.tag
                    if cand_tag.endswith('c') or cand_tag == 'c':
                        # Punctuation is always a meaningful stop
                        next_meaningful = cand
                        break
                    if cand_tag.endswith('w') or cand_tag == 'w':
                        cand_seg = cand.find('.//seg[@function="mrw"][@type="met"]')
                        if cand_seg is None:
                            cand_seg = cand.find('.//tei:seg[@function="mrw"][@type="met"]', namespace)
                        if get_word_text(cand, cand_seg).strip():
                            next_meaningful = cand
                            break
                        # else: empty placeholder — keep scanning

                add_trailing_space = False
                if next_meaningful is not None:
                    nm_tag = next_meaningful.tag.split('}')[-1] if '}' in next_meaningful.tag else next_meaningful.tag
                    if nm_tag.endswith('c') or nm_tag == 'c':
                        # Next is punctuation: only add space before opening brackets/parens.
                        # Closing punctuation (,  .  :  ;  !  ?  )) and apostrophes attach
                        # directly to the preceding word — the <c> element itself carries
                        # the trailing space.
                        nm_text = (next_meaningful.text or '').strip()
                        if nm_text and nm_text[0] in ('(', '[', '{'):
                            add_trailing_space = True
                    else:
                        # Next is a word element: always add a space, EXCEPT before a
                        # contraction/possessive suffix that starts with an apostrophe
                        # (e.g. 's, 've, 'll — those attach directly to the preceding word),
                        # or before an n't negation clitic (do + n't → don't).
                        nm_seg = next_meaningful.find('.//seg[@function="mrw"][@type="met"]')
                        if nm_seg is None:
                            nm_seg = next_meaningful.find('.//tei:seg[@function="mrw"][@type="met"]', namespace)
                        next_text = get_word_text(next_meaningful, nm_seg).strip()
                        if not next_text.startswith("'") and not next_text.startswith("n'"):
                            add_trailing_space = True

                sentence_text = word_text + (' ' if add_trailing_space else '')
                word_for_classification = word_text

                if not word_for_classification:
                    continue

                # ── Apostrophe / contraction suffixes ('s, 've, 'll, 't, 're) ──
                # Only merge genuine contractions (len > 1). A bare standalone
                # quote char (len == 1) falls through to the POS skip below.
                if word_for_classification.startswith("'") and len(word_for_classification) > 1:
                    # Remove any trailing space from the preceding word so that
                    # e.g. "Party " + "'s " becomes "Party's " not "Party 's "
                    if sentence_parts:
                        sentence_parts[-1] = sentence_parts[-1].rstrip(' ')
                    sentence_parts.append(sentence_text)
                    if current_words:
                        current_words[-1] = f"{current_words[-1]}{word_for_classification}"
                    continue

                # ── BNC negation clitic (n't) ────────────────────────────────
                # BNC stores contractions like "don't" / "can't" / "shouldn't"
                # as two separate <w> elements: the auxiliary/verb (e.g. "do")
                # and the negation clitic tagged XX0 (e.g. "n't").  Merge the
                # clitic back onto the preceding word so the reconstructed
                # sentence reads "don't" rather than "do n't".
                if word_for_classification.startswith("n'"):
                    if sentence_parts:
                        sentence_parts[-1] = sentence_parts[-1].rstrip(' ')
                    sentence_parts.append(sentence_text)
                    if current_words:
                        current_words[-1] = f"{current_words[-1]}{word_for_classification}"
                    else:
                        # Edge case: n't with no preceding word — keep as standalone
                        current_words.append(word_for_classification)
                        current_labels.append(label)
                    continue

                # POS-type elements are possessive markers (plural ' or singular 's).
                # They always attach to the preceding word — strip any trailing space
                # from the previous sentence part and merge into the preceding word entry.
                if word_type == 'POS':
                    if sentence_parts:
                        sentence_parts[-1] = sentence_parts[-1].rstrip(' ')
                    sentence_parts.append(sentence_text)
                    if current_words:
                        current_words[-1] = f"{current_words[-1]}{word_for_classification}"
                    continue

                if sentence_text:
                    sentence_parts.append(sentence_text)

                if is_classifiable_word(word_for_classification):
                    # Some BNC <w> elements are multi-word units (e.g. "per cent",
                    # "in answer to"). The sentence keeps them as-is, but words_only
                    # must list each whitespace-separated token separately so it stays
                    # 1:1 with the tokenizer's word segmentation (which splits on
                    # spaces). Each piece inherits the same metaphor label.
                    pieces = word_for_classification.split()
                    for piece in pieces:
                        if is_classifiable_word(piece):
                            current_words.append(piece)
                            current_labels.append(label)

            # ── Process punctuation elements ──────────────────────────────────
            elif elem_local == 'c':
                punct_text = elem.text if elem.text else ''
                # BNC opening quotation marks carry a trailing space in their
                # text content (e.g. '\u2018 ').  That space is a BNC word-
                # separator artefact — an opening quote must sit flush against
                # the first quoted word, so we strip it here.
                _OPENING_QUOTES = {'\u2018', '\u201c', '\u00ab', '\u2039'}  # ' " « ‹
                if punct_text.rstrip() in _OPENING_QUOTES:
                    punct_text = punct_text.rstrip()
                if punct_text:
                    sentence_parts.append(punct_text)
                # Don't add to words_only or word_labels
        
        if sentence_parts:  # Only add if we found content
            # Concatenate all parts as-is (no stripping, no rejoining with spaces)
            sentence = ''.join(sentence_parts)
            sentences.append(sentence)
            words_only.append(current_words)
            word_labels.append(current_labels)
    
    return sentences, words_only, word_labels

