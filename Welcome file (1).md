
At a high level, **EzArena just grabs the entire HTML of `https://www.heartharena.com/tierlist`, parses it with Beautiful Soup, and walks the DOM to build an in-memory dictionary keyed by HearthArena’s eight named tiers.** The scraper never touches a JSON or hidden API—everything comes straight out of the publicly rendered tier-list page. Below is a step-by-step walkthrough, annotated with the exact lines that do the work and a few notes about why each decision was made.

----------

## 1. Where the scrape happens

-   **Fetching the page**
    
    ```python
    html = urllib.urlopen('http://www.heartharena.com/tierlist').read()
    soup = BS(html, 'html.parser')
    
    ```
    
    These two lines in `arena.py` download the page with the standard‐library `urllib` and feed it to Beautiful Soup 4 for parsing ([GitHub](https://raw.githubusercontent.com/jdrusso/EzArena/master/arena.py "raw.githubusercontent.com")).  
    _The choice of `urllib.urlopen` is simple but adequate for a one-off grab; Python’s docs show that `urlopen()` returns a bytes object that can be `.read()` into memory_ ([Python documentation](https://docs.python.org/3/howto/urllib2.html?utm_source=chatgpt.com "HOWTO Fetch Internet Resources Using The urllib Package ...")).
    
-   **Locating the section for the player’s class**
    
    The scraper previously auto-detects the hero portrait (Rogue, Mage, etc.) and stores the literal class name string in `hero` (e.g., `"mage"`). It then narrows the soup to the `<div>` (or `<section>`) whose **`id` matches that class**:
    
    ```python
    tierlist = soup.find(id=hero)
    
    ```
    
    ([GitHub](https://raw.githubusercontent.com/jdrusso/EzArena/master/arena.py "raw.githubusercontent.com")).  
    `find(id=…)` is a canonical Beautiful Soup pattern for pinpointing a single element by its HTML `id` attribute ([crummy.com](https://www.crummy.com/software/BeautifulSoup/bs4/doc/?utm_source=chatgpt.com "Beautiful Soup 4.13.0 documentation - Crummy")).
    

----------

## 2. Understanding HearthArena’s markup

Inside each class block HearthArena groups cards by tier. Each tier is marked up with

```html
<ol class="tier good"> … <dt>Card Name 123</dt> …

```

-   `<ol>` uses the CSS class `tier X`, where `X` is the dash-separated tier name (`beyond-great`, `great`, `good`, …).
    
-   Each card appears in a `<dt>` (definition-term) element — per HTML spec, `<dt>` is legal inside definition/description lists (DL/DT/DD) ([MDN Web Docs](https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Elements/dt?utm_source=chatgpt.com "<dt>: The Description Term element - HTML - MDN Web Docs")).
    

That structure is why the scraper can rely solely on class names and `<dt>` tags; it never needs to parse scores or images.

----------

## 3. Walking the DOM and filling the data structure

The file defines an ordered list of the eight tiers:

```python
TIERS = ['beyond-great', 'great', 'good', 'above-average',
         'average', 'below-average', 'bad', 'terrible']

```

([GitHub](https://raw.githubusercontent.com/jdrusso/EzArena/master/arena.py "raw.githubusercontent.com"))

Then for every tier it executes:

```python
cards[tier] = []
for t in tierlist.findAll(class_="tier %s" % tier):
    for card in t.find('ol').findAll('dt'):
        if card.get_text() == u'\xa0':    # skip blank placeholders
            break
        cards[tier].append(card.get_text()[:-1])

```

([GitHub](https://raw.githubusercontent.com/jdrusso/EzArena/master/arena.py "raw.githubusercontent.com"))

Key points:

Step

What it does

Why it works

`findAll(class_='tier …')`

Returns every `<ol>` for that tier.

HearthArena sometimes splits a single tier across rarities, so there can be multiple `<ol>` blocks per tier.

Nested `find('ol').findAll('dt')`

Grabs each `<dt>` card entry.

`<dt>` = card name; the trailing colon & score are trimmed via `[:-1]`.

Unicode check `u'\xa0'`

Skips empty list entries that HearthArena pads with a non-breaking space.

Prevents blank strings from polluting the list.

The end result is **`cards` = `{ 'great': ['Fireball', 'Flamestrike', …], … }`** containing only strings.

----------

## 4. How the scraper uses the data later

Once the program OCRs the three draft options, it needs to match the noisy OCR text to the canonical names in `cards`. It compares every OCR result against every name in every tier using `difflib.SequenceMatcher().ratio()` to pick the best fuzzy match ([GitHub](https://raw.githubusercontent.com/jdrusso/EzArena/master/arena.py "raw.githubusercontent.com"), [Python documentation](https://docs.python.org/3/library/difflib.html?utm_source=chatgpt.com "difflib — Helpers for computing deltas — Python 3.13.5 documentation")). Cards are then ranked simply by iterating through `TIERS` in order and selecting the first match found — exploiting the fact that HearthArena presents tiers from strongest to weakest.

----------

## 5. Related files (but **not** part of scraping)

-   **`uploader.py`** just uploads the pre-processed card image to picpaste.com so the AWS OCR server can reach it; it never touches HearthArena ([GitHub](https://raw.githubusercontent.com/jdrusso/EzArena/master/uploader.py "raw.githubusercontent.com")).
    
-   **`pytesser.py`** and the GUI overlay code relate to OCR and user experience, not data acquisition.
    

----------

## 6. Why this approach works (and its limits)

-   **Pros**
    
    -   Zero dependency on a private API—anyone can replicate it with basic HTTP and HTML parsing.
        
    -   Resilient to minor layout tweaks because it anchors on semantic cues (`id=class`, `class="tier X"`, `<dt>`).
        
-   **Cons**
    
    -   **Full-page download every run.** A single change to HearthArena’s markup (e.g., switching `<dt>` to `<span>`) breaks the scraper.
        
    -   **No caching:** the scraper re-downloads the page for every draft, even though the tier list changes infrequently.
        
    -   **Fragile string trim:** removing the last character assumes the score is always exactly one trailing digit; multi-digit scores would truncate part of the name.
        
    -   **`urllib` & synchronous I/O:** modern code would use `requests` (already present in `requirements.txt`) or async fetches for readability and retries.
        

----------

## 7. Quick checklist if you need to adapt or modernize

1.  **Replace `urllib.urlopen`** with `requests.get(url, timeout=5)` and check `r.ok`.
    
2.  **Cache the response** locally and re-use it until HearthArena’s “Changelog” date changes.
    
3.  **Use CSS selectors** (`soup.select`) instead of nested `find*` calls for clarity.
    
4.  **Strip scores more robustly** with `card.get_text(strip=True).rsplit(' ', 1)[0]`.
    
5.  **Handle HTML changes gracefully** by asserting that each tier produced ≥ 1 card; log a warning if not.
    

----------

### TL;DR

_Download → parse → loop over eight tier names → scrape `<dt>` text → build a dict._ Everything lives in less than 20 lines of `arena.py`, and no other files participate in pulling data from HearthArena.
