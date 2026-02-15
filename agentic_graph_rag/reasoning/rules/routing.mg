% Query routing rules — declarative replacement for regex patterns in router.py
% Keywords map query tokens to categories

% Relation keywords (RU + EN)
keyword(/relation, "связ").
keyword(/relation, "отношен").
keyword(/relation, "соедин").
keyword(/relation, "relat").
keyword(/relation, "connect").
keyword(/relation, "link").
keyword(/relation, "between").
keyword(/relation, "между").

% Multi-hop keywords
keyword(/multi_hop, "цепочк").
keyword(/multi_hop, "путь").
keyword(/multi_hop, "сравн").
keyword(/multi_hop, "через").
keyword(/multi_hop, "chain").
keyword(/multi_hop, "path").
keyword(/multi_hop, "compar").
keyword(/multi_hop, "through").
keyword(/multi_hop, "affect").
keyword(/multi_hop, "влия").

% Global keywords
keyword(/global, "все").
keyword(/global, "кажд").
keyword(/global, "обзор").
keyword(/global, "список").
keyword(/global, "all").
keyword(/global, "every").
keyword(/global, "overview").
keyword(/global, "list").
keyword(/global, "summar").

% Temporal keywords
keyword(/temporal, "когда").
keyword(/temporal, "дата").
keyword(/temporal, "время").
keyword(/temporal, "истори").
keyword(/temporal, "when").
keyword(/temporal, "date").
keyword(/temporal, "timeline").
keyword(/temporal, "before").
keyword(/temporal, "after").
keyword(/temporal, "до").
keyword(/temporal, "после").

% Match: keyword must bind Word first, then query_contains checks it
match(Query, Category) :- keyword(Category, Word), query_contains(Query, Word).

% Tool mapping per category
tool_for(/simple, "vector_search").
tool_for(/relation, "cypher_traverse").
tool_for(/multi_hop, "cypher_traverse").
tool_for(/global, "full_document_read").
tool_for(/temporal, "temporal_query").

% Route: if any category matches, use its tool
route_to(Tool, Query) :- match(Query, Category), tool_for(Category, Tool).

% Default: no match → vector_search
route_to("vector_search", Query) :- current_query(Query), !match(Query, X).
