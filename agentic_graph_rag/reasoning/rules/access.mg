% Access control rules — role-based policies for RAG result filtering

% Role inheritance hierarchy
role_inherits(/admin, /analyst).
role_inherits(/analyst, /viewer).

% Recursive role resolution
has_role(User, Role) :- user_role(User, Role).
has_role(User, Parent) :- has_role(User, Child), role_inherits(Child, Parent).

% Permissions: role, action, resource type
permit(/viewer, /read, /public).
permit(/analyst, /read, /sensitive).
permit(/admin, /read, /pii).

% Deny overrides
deny(User, /write, /pii) :- has_role(User, X).

% Access decision: permit AND NOT deny
allowed(User, Action, ResType) :-
    has_role(User, Role), permit(Role, Action, ResType),
    !deny(User, Action, ResType).

% Enumerate possible actions and resource types for negation
all_actions(/read).
all_actions(/write).
all_resource_types(/public).
all_resource_types(/sensitive).
all_resource_types(/pii).

% Denied query audit trail (requires enumerated action/resource for sound negation)
denied_query(User, Action, ResType) :-
    has_role(User, Role), all_actions(Action), all_resource_types(ResType),
    !allowed(User, Action, ResType).
