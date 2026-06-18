import xml.etree.ElementTree as etree
import copy

def extract_xml_subset(xml_input_file, xml_output_file, number_of_sentences=50):
    # Parse the XML file
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
    
    # Create body
    body = etree.SubElement(text_elem, f'{{{ns_uri}}}body')
    
    # Create div1
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
    
    # Write the modified XML to the output file
    new_tree.write(xml_output_file, encoding='utf-8', xml_declaration=True)
    
    print(f"Extracted {len(selected_sentences)} sentences to {xml_output_file}")


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
            # Process word elements
            if elem.tag.endswith('w') or 'w' in elem.tag:
                # Check if this word has metaphor annotation
                seg_elem = elem.find('.//seg[@function="mrw"][@type="met"]')
                if seg_elem is None:
                    seg_elem = elem.find('.//tei:seg[@function="mrw"][@type="met"]', namespace)
                
                # Get the word text
                if seg_elem is not None:
                    # For metaphor words, seg element contains the word
                    seg_text = seg_elem.text if seg_elem.text else ''
                    
                    # Check next element to decide if we need trailing space
                    # Add space only if next element's text starts with alphanumeric or opening bracket
                    # If next element's text starts with whitespace, preserve it (don't add trailing space)
                    next_elem = children[i + 1] if i + 1 < len(children) else None
                    add_trailing_space = False
                    if next_elem is not None:
                        # Check if next element is also a metaphor word
                        next_seg = next_elem.find('.//seg[@function="mrw"][@type="met"]')
                        if next_seg is None:
                            next_seg = next_elem.find('.//tei:seg[@function="mrw"][@type="met"]', namespace)
                        
                        if next_seg is not None and next_seg.text:
                            # Next is a metaphor word, check its seg text
                            next_text = next_seg.text
                        elif next_elem.text:
                            # Next is a regular word, check its parent text
                            next_text = next_elem.text
                        else:
                            next_text = ''
                        
                        if next_text:
                            # If next text starts with whitespace, preserve it (don't add trailing space)
                            if next_text[0].isspace():
                                add_trailing_space = False
                            # If next text starts with alphanumeric or opening bracket, add trailing space
                            elif next_text[0].isalnum() or next_text[0] in ['(', '[', '{']:
                                add_trailing_space = True
                            # If next text starts with punctuation, don't add trailing space
                            else:
                                add_trailing_space = False
                    
                    sentence_text = seg_text + (' ' if add_trailing_space else '')
                    word_for_classification = seg_text.strip()
                    label = 1
                else:
                    # For non-metaphor words, use parent element text as-is
                    # Skip if parent element is only whitespace (XML formatting)
                    if elem.text and not elem.text.strip():
                        continue
                    # Extract only the actual word (before any newlines in the text)
                    full_text = elem.text if elem.text else ''
                    sentence_text = full_text.split('\n')[0] if '\n' in full_text else full_text
                    word_for_classification = sentence_text.strip() if sentence_text else ''
                    label = 0
                
                # Skip if the word text is only whitespace (XML formatting)
                if not word_for_classification or not sentence_text:
                    continue
                
                # Add to sentence as-is (preserving original spacing)
                if sentence_text:
                    sentence_parts.append(sentence_text)
                
                # Add to words_only only if it's a real word (not punctuation)
                if word_for_classification and word_for_classification not in [',', '.', ':', ';', '!', '?', '"', "'", '""', '"', '(', ')', '[', ']', '{', '}']:
                    current_words.append(word_for_classification)
                    current_labels.append(label)
            
            # Process punctuation elements
            elif elem.tag.endswith('c') or 'c' in elem.tag:
                punct_text = elem.text if elem.text else ''
                # Add to sentence as-is (preserving original spacing)
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


def display_extraction_results(sentences, words_only, word_labels, max_display=10):
    """
    Display the extraction results for verification.
    
    Args:
        sentences (list): List of sentence strings (with punctuation)
        words_only (list): List of word lists (only words, no punctuation)
        word_labels (list): List of label lists (corresponding to words_only)
        max_display (int): Maximum number of sentences to display
    """
    print("=" * 80)
    print("VUAMC DATA EXTRACTION RESULTS")
    print("=" * 80)
    print(f"Total sentences extracted: {len(sentences)}")
    print(f"Total word lists extracted: {len(words_only)}")
    print(f"Total label lists extracted: {len(word_labels)}")
    print()
    
    # Display statistics (only count word labels, not punctuation)
    total_words = sum(len(labels) for labels in word_labels)
    total_metaphors = sum(sum(labels) for labels in word_labels)
    print(f"Total words (with labels): {total_words}")
    print(f"Total metaphors (label=1): {total_metaphors}")
    if total_words > 0:
        print(f"Metaphor rate: {total_metaphors/total_words:.2%}")
    print()
    
    # Display first few sentences
    display_count = min(max_display, len(sentences))
    print(f"First {display_count} sentences with detailed breakdown:")
    print("-" * 80)
    
    for i in range(display_count):
        print(f"\nSentence {i+1}:")
        print(f"  Full text (with punctuation): {sentences[i]}")
        print(f"  Words only: {words_only[i]}")
        print(f"  Word labels: {word_labels[i]}")
        
        # Show which words are metaphors
        metaphor_words = [words_only[i][j] for j, label in enumerate(word_labels[i]) if label == 1]
        if metaphor_words:
            print(f"  Metaphor words: {metaphor_words}")
        else:
            print("  Metaphor words: None")
    
    print("\n" + "-" * 80)
    print(f"Showing {display_count} of {len(sentences)} sentences")
    print("Full results saved to extracted_data.txt")
    print("=" * 80)


if __name__ == "__main__":
    # Test the VUAMC data extraction
    xml_file = "data/VUAMC/2541/VUAMC_first_100_sentences.xml"
    
    sentences, words_only, word_labels = extract_vuamc_data(xml_file)
    display_extraction_results(sentences, words_only, word_labels, max_display=10)
    
    # Save to a file for reference
    output_file = "data/VUAMC/2541/extracted_data.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("VUAMC Data Extraction Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total sentences: {len(sentences)}\n")
        total_words = sum(len(labels) for labels in word_labels)
        total_metaphors = sum(sum(labels) for labels in word_labels)
        f.write(f"Total words: {total_words}\n")
        f.write(f"Total metaphors: {total_metaphors}\n")
        if total_words > 0:
            f.write(f"Metaphor rate: {total_metaphors/total_words:.2%}\n\n")
        
        for i in range(len(sentences)):
            f.write(f"Sentence {i+1}:\n")
            f.write(f"Full text: {sentences[i]}\n")
            f.write(f"Words only: {words_only[i]}\n")
            f.write(f"Word labels: {word_labels[i]}\n")
            metaphor_words = [words_only[i][j] for j, l in enumerate(word_labels[i]) if l == 1]
            f.write(f"Metaphor words: {metaphor_words}\n")
            f.write("-" * 80 + "\n")
    
    print(f"\nExtraction results saved to: {output_file}")
