from query import search

# unfiltered vs scoped
for content, company, subject, url, source in search("data integrity expectations", source="MHRA"):
    print(source, "|", company[:50])