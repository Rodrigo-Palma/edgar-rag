from edgar_rag.edgar.parse import html_to_text, split_into_sections

FILING = """
<html><body>
<p>Item 1. Business</p>
<p>{business}</p>
<p>Item 1A. Risk Factors</p>
<p>{risks}</p>
</body></html>
""".format(business="We sell things. " * 30, risks="Things may stop selling. " * 30)


def test_drops_markup_and_keeps_the_prose():
    text = html_to_text(
        "<html><body><script>ignore()</script><p>Hello&nbsp;world</p></body></html>"
    )

    assert "ignore()" not in text
    assert "Hello world" in text


def test_splits_a_filing_on_its_item_headings():
    sections = split_into_sections(html_to_text(FILING))

    assert [section.item for section in sections] == ["Item 1", "Item 1A"]
    assert "We sell things." in sections[0].text
    assert "Things may stop selling." in sections[1].text


def test_a_table_of_contents_entry_is_not_a_section():
    toc = "Item 1. Business 3\nItem 1A. Risk Factors 12\n" + FILING

    sections = split_into_sections(html_to_text(toc))

    # The headings appear twice, once as a line of the contents and once for real
    assert [section.item for section in sections] == ["Item 1", "Item 1A"]


def test_falls_back_to_one_section_when_no_heading_is_found():
    sections = split_into_sections("A filing with no item headings at all.")

    assert len(sections) == 1
    assert sections[0].item == "Full filing"
