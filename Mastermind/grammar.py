from primitives import *
from utils import *
import numpy as np
import time
from math import log, exp
from itertools import combinations, permutations, product

# Add file-level docstring at the top
"""
Grammar implementation for the Mastermind game.
This module provides functionality to create and manipulate grammatical expressions
for representing Mastermind game logic, including code generation and possibility checking.
"""

def get_all_codes(n_positions, n_colors):
    """Generate all possible code combinations for the game.
    
    Args:
        n_positions: Number of positions in the code
        n_colors: Number of possible colors (1 to n_colors)
    
    Returns:
        List of all possible code combinations
    """
    digs = [i for i in range(1,n_colors+1)]
    codes = list(product(digs, repeat=n_positions))
    return [List(*[Color(y) for y in x]) for x in codes]



def find_mastermind_possibilities(codes, query, feedback):
    """Find all possible codes that are consistent with the given query and feedback.
    
    Args:
        codes: List of all possible codes
        query: The guessed code
        feedback: Tuple of (correct positions, correct colors)
    
    Returns:
        List of valid combinations that match the feedback
    """
    query_colors = set(query)


    valid_combs = []

    for code in codes:
        consistent = get_single_likelihood(query,  (code, feedback))

        if consistent:
            comb = []

            for i in range(len(code)):
                val = code[i]
                comb.append((i, code[i].execute({}), True))

            valid_combs.append(comb)

    return valid_combs



def possibility_to_expression(possibility, grammar, var):
    if len(possibility) == 0:
        return None

    var_type, var_name = var
    expressions = []

    for position, color, in_set in possibility:
        pos_rule = grammar.rules[Pos]["primitives"][grammar.rules[Pos]["primitives"].index((Number(position), [], None))]
        color_rule = grammar.rules[Elem]["primitives"][grammar.rules[Elem]["primitives"].index((Color(color), [], None))]
        equals_rule = grammar.rules[Bool]["primitives"][grammar.rules[Bool]["primitives"].index((Equals, [Elem, Elem], None))]
        var_rule = grammar.rules[var_type]["primitives"][grammar.rules[var_type]["primitives"].index((Var(var_name), [], None))]

        pos_log_prob = log(grammar.rules[Pos]["probabilities"][grammar.rules[Pos]["primitives"].index(pos_rule)])
        color_log_prob = log(grammar.rules[Elem]["probabilities"][grammar.rules[Elem]["primitives"].index(color_rule)])
        equals_log_prob = log(grammar.rules[Bool]["probabilities"][grammar.rules[Bool]["primitives"].index(equals_rule)])
        var_log_prob = log(grammar.rules[var_type]["probabilities"][grammar.rules[var_type]["primitives"].index(var_rule)])

        PosPrim = Primitive(Number(position), Pos, [], log_probability=pos_log_prob, size=1)
        ColorPrim = Primitive(Color(color), Elem, [], log_probability=color_log_prob, size=1)
        VarPrim = Primitive(Var(var_name), var_type, [], log_probability=var_log_prob, size=1)  
        IndexPrim = Primitive(Index, Elem, [VarPrim, PosPrim], log_probability=pos_log_prob + pos_log_prob, size=1 + VarPrim.size + PosPrim.size)

        if in_set:
            expressions.append(Primitive(Equals, Bool, [IndexPrim, ColorPrim], log_probability=equals_log_prob + IndexPrim.log_probability + ColorPrim.log_probability, size=1 + IndexPrim.size + ColorPrim.size))
        else:
            EqualsPrim = Primitive(Equals, Bool, [IndexPrim, ColorPrim], log_probability=equals_log_prob + IndexPrim.log_probability + ColorPrim.log_probability, size=1 + IndexPrim.size + ColorPrim.size)
            not_rule = grammar.rules[Bool]["primitives"][grammar.rules[Bool]["primitives"].index((Not, [Bool], None))]
            not_log_prob = log(grammar.rules[Bool]["probabilities"][grammar.rules[Bool]["primitives"].index(not_rule)])
            expressions.append(Primitive(Not, Bool, [EqualsPrim], log_probability=not_log_prob + EqualsPrim.log_probability, size=1 + EqualsPrim.size))

    # Combine expressions using And
    combined_expr = expressions[0]
    for expr in expressions[1:]:
        and_rule = grammar.rules[Bool]["primitives"][grammar.rules[Bool]["primitives"].index((And, [Bool, Bool], None))]
        and_log_prob = log(grammar.rules[Bool]["probabilities"][grammar.rules[Bool]["primitives"].index(and_rule)])
        combined_expr = Primitive(And, Bool, [combined_expr, expr], log_probability=and_log_prob + combined_expr.log_probability + expr.log_probability, size=1 + combined_expr.size + expr.size)

    return combined_expr




def possibilities_to_expression(possibilities, grammar, var=(Array, "x0"), simplify=True):
    if len(possibilities) == 0:
        return None
    elif len(possibilities) == 1:
        return possibility_to_expression(possibilities[0], grammar, var)
    else:

        if simplify:
            disjoint_poss = []
            for poss in possibilities:
                n_true = sum([p[2]*1 for p in poss])
                n_false = len(poss) - n_true

                disjoint_poss.append([p for p in poss if p[2]])

            expressions = [possibility_to_expression(p, grammar, var) for p in disjoint_poss] 

        else:

            expressions = [possibility_to_expression(p, grammar, var) for p in possibilities]
    

        xor_rule = grammar.rules[Bool]["primitives"][grammar.rules[Bool]["primitives"].index((Xor, [Bool, Bool], None))]
        xor_log_prob = log(grammar.rules[Bool]["probabilities"][grammar.rules[Bool]["primitives"].index(xor_rule)])

        def build_xor_tree(expressions):
            if len(expressions) == 1:
                return expressions[0]

        def combine_xor(expressions):
            if len(expressions) == 1:
                return expressions[0]

            mid = len(expressions) // 2
            left = combine_xor(expressions[:mid])
            right = combine_xor(expressions[mid:])

            return Primitive(Xor, Bool, [left, right], log_probability=xor_log_prob + left.log_probability + right.log_probability, size=1 + left.size + right.size)

        return combine_xor(expressions)




def find_disjunctive_representation(possibilities, grammar, simplify=True):

    var = (Array, "x0")
    grammar.add_rule(var[0], Var(var[1]), [], p=1.0)

    lambda_rule = grammar.rules[ArrayFunc]["primitives"][grammar.rules[ArrayFunc]["primitives"].index((Lambda, [Bool], [Array]))]
    statement_rule = grammar.rules[Expr]["primitives"][grammar.rules[Expr]["primitives"].index((Statement, [ArrayFunc], None))]

    lambda_prob = log(grammar.rules[ArrayFunc]["probabilities"][grammar.rules[ArrayFunc]["primitives"].index(lambda_rule)])

    possibilities_expr = possibilities_to_expression(possibilities, grammar, var, simplify=simplify)
    LambdaPrim = Primitive(Lambda, ArrayFunc, children=[possibilities_expr], bvs=[var], 
                log_probability=lambda_prob + possibilities_expr.log_probability, 
                size = 1 + possibilities_expr.size)
    grammar.remove_rule(var[0], Var(var[1]))

    return Primitive(Statement, Expr, [LambdaPrim],
             log_probability=LambdaPrim.log_probability, size=1+LambdaPrim.size)


def get_filter_likelihood(hypothesis, options,
                        test,true_lkhds, noise=1e-3, 
                        likelihood_limit=float("-inf")):


    guess, feedback = test
    h_lkhd = 0
    n_options = len(options)

    h_evals = []

    for k in range(n_options):

        true_lkhd = true_lkhds[k]
        code = options[k]
        h_eval = 1*hypothesis.evaluate([code])
        h_evals.append(h_eval)

        if (h_lkhd < likelihood_limit) or  (h_eval < true_lkhd):
            return float("-inf"), None
        elif ((k > 0) and 
                ((h_lkhd * (n_options/(k+1))) < likelihood_limit * 1.5)):
            return float("-inf"), None
        elif true_lkhd == h_eval:
            h_lkhd += log(1-noise)
        else:
            h_lkhd += log(noise)

    return h_lkhd, h_evals




class Grammar:
    """A class representing a probabilistic context-free grammar.
    
    This grammar is used to generate and manipulate logical expressions
    for the Mastermind game.
    """
    
    def __init__(self):
        """Initialize an empty grammar with no rules."""
        self.rules = {}



    def add_rule(self, parent_type, constructor, child_types, p=1.0, bv_types=None):
        """Add a production rule to the grammar.
        
        Args:
            parent_type: Type of the parent node
            constructor: Constructor function for the node
            child_types: List of types for child nodes
            p: Probability weight for this rule
            bv_types: Bound variable types (for Lambda expressions)
        """
        if parent_type not in self.rules:
            self.rules[parent_type] = {"primitives": [], 'weights':[], 'probabilities':[]}
        



        self.rules[parent_type]["primitives"].append((constructor, child_types,
                                                         bv_types))
        self.rules[parent_type]["weights"].append(p)

        weights = self.rules[parent_type]["weights"]
        self.rules[parent_type]["probabilities"] = [p / sum(weights) for p in weights]





    def remove_rule(self, parent_type, constructor, child_types=None, bv_types=None):
        if parent_type in self.rules:
            for i, (prim, ct, bts) in enumerate(self.rules[parent_type]["primitives"]):
                if ((prim == constructor) and ((bv_types == None) or (bts == bv_types)) and
                    ((child_types == None) or (ct == child_types))):
                    del self.rules[parent_type]["primitives"][i]
                    del self.rules[parent_type]["weights"][i]
                    break  # Assuming each constructor appears at most once for each parent type

            weights = self.rules[parent_type]["weights"]
            self.rules[parent_type]["probabilities"] = [p / sum(weights) for p in weights] if weights else []


    def __repr__(self):
        rules_str = []
        for parent_type, data in self.rules.items():
            for i, (constructor, child_types, bv_type) in enumerate(data["primitives"]):
                probability = data["probabilities"][i]
                constructor_name = constructor.__name__ if hasattr(constructor, '__name__') else str(constructor)

                child_types_str = ""
                if len(child_types) > 0:
                    child_types_str = f"({', '.join([t.__name__ for t in child_types])})"

                rules_str.append(f"{parent_type.__name__} -> {constructor_name}{child_types_str}, p={probability:.2f}")
        return '\n'.join(rules_str)




def remove_top(expr):

    if not expr.children:
        return expr

    same_type_children = [child for child in expr.children if child.primitive_type == expr.primitive_type]

    if not same_type_children:
        return expr  

    return random.choice(same_type_children)

def add_top(expr, grammar, env=None):

    if env is None:
        env = {}

    # Filter primitives to include only those that can have 'expr' as a child
    valid_primitives = [rule for rule in grammar.rules[expr.primitive_type]["primitives"] if\
                         expr.primitive_type in rule[1] and expr.primitive != Lambda]

    if not valid_primitives:
        return expr  # Return the original expression if no valid primitives are found

    new_primitive, new_children_types, bv_types = random.choice(valid_primitives)
    new_log_prob = log(random.choice(grammar.rules[expr.primitive_type]["probabilities"]))


    expr_position = random.randint(0, len(new_children_types) - 1)

    new_children = []
    for i, child_type in enumerate(new_children_types):
        if i == expr_position:
            new_children.append(expr)
        else:
            new_child = sample(grammar, child_type, copy.deepcopy(env))
            new_children.append(new_child)

    updated_log_prob = new_log_prob + sum(child.log_probability for child in new_children)
    updated_size = 1 + sum(child.size for child in new_children)

    new_expr = Primitive(new_primitive, expr.primitive_type, 
                new_children, log_probability=updated_log_prob, size=updated_size,
                bvs=expr.bvs)



    return new_expr


def sample(grammar, parent_type=Expr, env=None):
    if parent_type not in grammar.rules:
        return None

    if env is None:
        env = {}

    primitives = grammar.rules[parent_type]["primitives"]
    probabilities = grammar.rules[parent_type]["probabilities"]

    constructor, child_types, bv_types = random.choices(primitives, weights=probabilities, k=1)[0]
    chosen_prob = log(probabilities[primitives.index((constructor, child_types, bv_types))])

    if constructor == Lambda:
        bvs = []
        var_names = []

        for bv_type in bv_types:

            var_name = f"x{len(env)}"
            env[var_name] = bv_type
            grammar.add_rule(bv_type, Var(var_name), [], p=1.0)
            bvs.append((bv_type, var_name))

        body = sample(grammar, child_types[0], env)

        for bv_type, var_name in bvs:
            del env[var_name]
            grammar.remove_rule(bv_type, Var(var_name))
        #lambda_instance = Lambda([var_name for _, var_name in bvs], body)
        return Primitive(Lambda, parent_type, bvs=bvs, 
                    children=[body], log_probability=chosen_prob + body.log_probability, size = 1+body.size)

    if len(child_types) > 0:
        children = []
        children_log_probs = 0
        children_size = 0
        for child_type in child_types:
            child = sample(grammar, child_type, env)
            children.append(child)
            children_log_probs += child.log_probability
            children_size += child.size
        return Primitive(constructor, parent_type, children, log_probability=chosen_prob + children_log_probs, size=1+children_size)

    return Primitive(constructor, parent_type, log_probability=chosen_prob)





def resample_random_subtree(expr, grammar, env=None, depth=0,
             p_resample_node=0.5, p_add_or_remove=0.2):



    if env is None:
        env = {}

    if not expr.children:
        return sample(grammar, expr.primitive_type, env)

    total_size = expr.size

    if random.random() < p_resample_node:
        if (random.random() < p_add_or_remove) and (expr.primitive_type in [Bool]):

            if random.random() < 1/3:
                return add_top(expr, grammar, env)
            else:
                return remove_top(expr)
        else:
            return sample(grammar, expr.primitive_type, env)
    else:

        if (expr.primitive == Lambda) and expr.bvs:
            for bv_type, var_name in expr.bvs:
                env[var_name] = bv_type
                grammar.add_rule(bv_type, Var(var_name), [], p=1.0)



        child_idx = random.randint(0, len(expr.children) - 1)

        new_child = resample_random_subtree(expr.children[child_idx], grammar, env, depth+1)
        

        new_children = expr.children[:child_idx] + [new_child] + expr.children[child_idx + 1:]

        updated_log_prob = expr.log_probability + new_child.log_probability - expr.children[child_idx].log_probability
        updated_size = expr.size +  new_child.size - expr.children[child_idx].size

        if (expr.primitive == Lambda) and expr.bvs:
            for bv_type, var_name in expr.bvs:
                del env[var_name]
                grammar.remove_rule(bv_type, Var(var_name))

        return Primitive(expr.primitive, expr.primitive_type, 
            new_children, log_probability=updated_log_prob,
             size=updated_size, bvs=expr.bvs)






def make_grammar(n_positions, n_colors):
    """Create a grammar for the Mastermind game with given parameters.
    
    Args:
        n_positions: Number of positions in the code
        n_colors: Number of possible colors
        
    Returns:
        Grammar object with rules for generating Mastermind expressions
    """
    grammar = Grammar()

    grammar.add_rule(Expr, Statement, [ArrayFunc], p=1)
    grammar.add_rule(ArrayFunc, Lambda, [Bool], 
                               p=1, bv_types=[Array])


    positions = []
    for i in range(0,n_positions):
        grammar.add_rule(Pos, Number(i), [], p=1)


    for i in range(1,n_colors+1):
        grammar.add_rule(Elem, Color(i), [], p=1)

    grammar.add_rule(Elem, Index, [Array, Pos], p=1)

    grammar.add_rule(BoolFunc, Lambda, [Bool], p=1, bv_types=[Elem])
    grammar.add_rule(Bool, Equals, [Elem, Elem], p=1)

    grammar.add_rule(Bool, Exists, [BoolFunc, Array], p=0.1)
    grammar.add_rule(Bool, ExistsUnique, [BoolFunc, Array], p=0.1)
    grammar.add_rule(Bool, Forall, [BoolFunc, Array], p=0.1)

    grammar.add_rule(Bool, In, [Elem, Array], p=1)
    grammar.add_rule(Bool, And, [Bool, Bool], p=0.125)
    grammar.add_rule(Bool, Or, [Bool, Bool], p=0.125)
    grammar.add_rule(Bool, Xor, [Bool, Bool], p=0.125)
    grammar.add_rule(Bool, Not, [Bool], p=0.25) 


    return grammar



if __name__ == "__main__":
    # Example usage of the grammar system
    code_length = 4
    n_colors = 4

    # Generate all possible codes
    codes = get_all_codes(code_length, n_colors)
    
    # Create grammar and sample an expression
    grammar = make_grammar(code_length, n_colors)
    expr = sample(grammar)
    