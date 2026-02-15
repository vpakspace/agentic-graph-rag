% Graph reasoning rules — inference over PhraseNode/PassageNode graph
% External predicates: phrase_node, edge, mentioned_in (from Neo4j)

% Transitive closure: reachable up to depth 5
reachable(X, Y, 1) :- edge(X, R, Y).
reachable(X, Z, D) :- reachable(X, Y, D1), edge(Y, R, Z),
    D = fn:plus(D1, 1), D < 5.

% Common neighbor: two different nodes share a neighbor
common_neighbor(A, B, Neighbor) :- edge(A, R1, Neighbor), edge(B, R2, Neighbor), A != B.

% Evidence: entity mentioned in a passage
evidence(Entity, PassageId) :- mentioned_in(Entity, PassageId, Text).
