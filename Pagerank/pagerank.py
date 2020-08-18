import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    linked_pages=corpus[page]
    page_rank=dict()
    if len(linked_pages) ==0 :
        for i in corpus:
            page_rank[i]=float(1/len(corpus))
    else:
        for i in corpus:
            page_rank.update({i:float((1-damping_factor)/len(corpus))})
            
        
        for i in linked_pages:
            
            
            page_rank.update({i:float((damping_factor)/len(linked_pages))})
        
    return page_rank    


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    
    page_rank=dict()
    sample=None
    
    for sample in corpus:
        page_rank[sample]=0
    for i in range(n):
        if sample is None:
            sample=random.choices(list(corpus.keys()),k=1)[0]
        else:
            model_copy=transition_model(corpus, sample, damping_factor)
            population,weight=zip(*model_copy.items())
            
            sample=random.choices(population,weights=weight,k=1)[0]
        page_rank[sample] += 1/n
    return page_rank   
            
    
    
    
    
    
    


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    
    pagerank=dict()
    finalrank=dict()
    rep=True

    N=len(corpus)
    for i in corpus:
        pagerank[i]=1/N
    while rep:
        
        for page in pagerank:
            pr = 0

            for inside in corpus:
                if not corpus[inside]:
                    pr += pagerank[inside] / len(corpus)
                
                if page in corpus[inside]:
                    pr += pagerank[inside] / len(corpus[inside])
               
            finalrank[page] = (1 - damping_factor)/N + damping_factor*pr

        rep = False
        for pag in pagerank:
                
            if abs(finalrank[pag]-pagerank[pag]) > 0.001:
                rep = True
            pagerank[pag]=finalrank[pag]

    return pagerank
    
                    
               
        
                
                
        
    


if __name__ == "__main__":
    main()
