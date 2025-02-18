import random
import copy

class Int:
    pass

class Pos:
    pass



class Elem:
    pass

class Alt:
    pass

class Array:
    pass

class Code:
    pass


class Positions:
    pass

class Op:
    pass

class Expr:
    pass

class Bool:
    pass

class Function:
    pass

class ArrayVar:
    pass

class ArrayFunc:
    pass

class IntVar:
    pass

class IntFunc:
    pass

class BoolVar:
    pass

class BoolFunc:
    pass






class Primitive:
    def __init__(self, primitive, primitive_type, children=[], bvs=None, log_probability=float("-inf"), size=1):
        self.primitive = primitive
        self.primitive_type = primitive_type
        self.children = children
        self.bvs = bvs

        self.params = []
        if self.bvs:
            self.params = [var_name for _, var_name in self.bvs]
        
        self.log_probability = log_probability
        self.size = size


    def __repr__(self):
        if self.primitive == Lambda:
            return f"{repr(self.primitive(self.params, self.children[0]))}"

        elif self.children:
            return f"{repr(self.primitive(*self.children))}"
        else:
            return repr(self.primitive)



    def execute(self, env):

        if self.primitive == Lambda:
            return self.primitive(self.params, self.children[0]).execute(env)
        elif callable(self.primitive):
            instance = self.primitive(*self.children)
            if hasattr(instance, 'execute'):
                return instance.execute(env)
            elif hasattr(instance, 'evaluate'):
                return instance.evaluate(env)
        elif hasattr(self.primitive, 'execute'):
            return self.primitive.execute(env)
        elif hasattr(self.primitive, 'evaluate'):
            return self.primitive.evaluate(env)
        else:
            raise TypeError(f"Primitive does not have an executable method: {type(self.primitive)}")

    def evaluate(self, env={}):
        return self.execute(env)



class Statement:
    def __init__(self, expression):
        self.expression = expression

    def __repr__(self):
        return f"{repr(self.expression)}"

    def __eq__(self, other):
        return isinstance(other, Statement) and other.expression==self.expression

    def evaluate(self, args=[]):

        if len(args) > 0:
            applied_expr = Apply(self.expression, *args)
            return applied_expr.execute({})
        else:
            return self.expression.execute({})


class Lambda:
    def __init__(self, params, body):
        self.params = params
        self.body = body
        self.arity = len(params)

    def __repr__(self):
        return f"lambda {', '.join(self.params)}: ({repr(self.body)})"

    def __eq__(self, other):
        return isinstance(other, Lambda) and (self.params == other.params) and (self.body == other.body)

    def execute(self, env):
        return Closure(self.params, self.body, env)




class Apply:
    def __init__(self, lambda_expr, *args):
        self.lambda_expr = lambda_expr
        self.args = args

    def __repr__(self):
        return f"{repr(self.lambda_expr)} ({', '.join(map(repr, self.args))})"

    def __eq__(self, other):
        return isinstance(other, Apply) and (self.lambda_expr == other.lambda_expr) and (self.args == other.args)

    def execute(self, env):
        #closure = self.lambda_expr.execute(copy.deepcopy(env))
        closure = self.lambda_expr.execute(env.copy())

        arg_thunks = [Thunk(arg, env) for arg in self.args]
        return closure.apply(arg_thunks)

class Closure:
    def __init__(self, params, body, env):
        self.params = params
        self.body = body
        self.env = env
        self.arity = len(params)

    def apply(self, args):
        if len(args) != self.arity:
            raise ValueError(f"Expected {self.arity} arguments, got {len(args)}")
        new_env = self.env.copy()
        for param, arg in zip(self.params, args):
            new_env[param] = arg
        return self.body.execute(new_env)


class Thunk:
    def __init__(self, body, env):
        self.body = body
        self.env = env

    def unthunk(self):
        return self.body.execute(self.env)

class Var:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"{self.name}"

    def __eq__(self, other):
        return isinstance(other, Var) and self.name == other.name

    def execute(self, env):
        thunk = env[self.name]
        return thunk.unthunk()



class Number:
    def __init__(self, value):
        self.value = value
    def __repr__(self):
        return f"{repr(self.value)}"

    def __hash__(self):
        return self.value

    def __eq__(self, other):
        return isinstance(other, Number) and self.value == other.value
    def execute(self, env):
        return self.value


class Color:
    def __init__(self, value):
        self.value = value
    def __repr__(self):
        return f"{repr(self.value)}"

    def __hash__(self):
        return self.value

    def __eq__(self, other):
        return isinstance(other, Color) and self.value == other.value
        
    def execute(self, env):
        return self.value




class Plus:
    def __init__(self, v1, v2):
        self.v1 = v1
        self.v2 = v2
    def __repr__(self):
        return f"({repr(self.v1)} + {repr(self.v2)})"
    def __eq__(self, other):
        return isinstance(other, Plus) and self.v1 == other.v1 and self.v2 == other.v2
    def execute(self, env):
        return self.v1.execute(env) + self.v2.execute(env)




class Count:
    def __init__(self, A, x):
        self.A = A
        self.x = x
    def __repr__(self):
        return f"Count({repr(self.A)}, {repr(self.x)})"
    def __eq__(self, other):
        return isinstance(other, Count) and self.A == other.A and self.x == other.x
    def execute(self, env):
        return self.A.execute(env).count(self.x.execute(env))



class List:
    def __init__(self, *elements):
        self.elements = list(elements)

    def __repr__(self):
        return f"{repr(self.elements)}"

    def __len__(self):
        return len(self.elements)

    def __eq__(self, other):
        return isinstance(other, List) and self.elements == other.elements

    def __getitem__(self, index):
        return self.elements[index]


    def __iter__(self):
        return iter(self.elements)

    def execute(self, env):
        return [el.execute(env) for el in self.elements]



class Index:
    def __init__(self, L, idx):
        self.L = L
        self.idx = idx

    def __repr__(self):
        return f"{repr(self.L)}[{repr(self.idx)}]"


    def execute(self, env):
        L = self.L.execute(env)
        idx = self.idx.execute(env)

        if not isinstance(idx, int) or (idx >= len(L)):
            return Number(None)
        else:
            return L[idx]



class Map:
    def __init__(self, lambda_expr, lst):
        self.lambda_expr = lambda_expr
        self.lst = lst

    def __repr__(self):
        return f"Map({repr(self.lambda_expr)}, {repr(self.lst)})"

    def __eq__(self):
        return (isinstance(other, Map) and self.lambda_expr == other.lambda_expr and 
                        self.lst == other.lst)


    def execute(self, env):
        lst_val = self.lst.execute(env)
        return [Apply(self.lambda_expr, Number(el)).execute(env) for el in lst_val]


class Wrapper:
    def __init__(self, A):
        self.A = A

    def __repr__(self):
        return f"({repr(self.A)})"


    def __eq__(self, other):
        return isinstance(other, Wrapper) and self.A == other.A 

    def execute(self, env):
        return (self.A.execute(env))

class And:
    def __init__(self, A, B):
        self.A = A
        self.B = B

    def __repr__(self):
        return f"({repr(self.A)} & {repr(self.B)})"


    def __eq__(self, other):
        return isinstance(other, And) and self.A == other.A and self.B == other.B

    def execute(self, env):
        return (self.A.execute(env) and self.B.execute(env))



class Xor:
    def __init__(self, A, B):
        self.A = A
        self.B = B

    def __repr__(self):
        return f"({repr(self.A)} ^ {repr(self.B)})"


    def __eq__(self, other):
        return isinstance(other, Xor) and self.A == other.A and self.B == other.B

    def execute(self, env):

        return (self.A.execute(env) ^ self.B.execute(env))



class Or:
    def __init__(self, A, B):
        self.A = A
        self.B = B

    def __repr__(self):
        return f"({repr(self.A)} | {repr(self.B)})"


    def __eq__(self, other):
        return isinstance(other, Or) and self.A == other.A and self.B == other.B

    def execute(self, env):

        return (self.A.execute(env) or self.B.execute(env))


class Not:
    def __init__(self, A):
        self.A = A

    def __repr__(self):
        return f"!({repr(self.A)})"


    def __eq__(self, other):
        return isinstance(other, Not) and self.A == other.A

    def execute(self, env):
        return not (self.A.execute(env))



class Equals:
    def __init__(self, A, B):
        self.A = A
        self.B = B

    def __repr__(self):
        return f"({repr(self.A)} == {repr(self.B)})"

    def __eq__(self, other):
        return isinstance(other, Equals) and self.A == other.A and self.B == other.B

    def execute(self, env):

        return (self.A.execute(env) == self.B.execute(env))


class Implies:
    def __init__(self, A, B):
        self.A = A
        self.B = B

    def __repr__(self):
        return f"{repr(self.A)} => {repr(self.B)}"

    def __eq__(self, other):
        return isinstance(other, Implies) and self.A == other.A and self.B == other.B

    def execute(self, env):

        return (not (self.A.execute(env))) or self.B.execute(env)



class In:
    def __init__(self, element, list):
        self.element = element
        self.list = list

    def execute(self, env):
        L = self.list.execute(env)
        v = self.element.execute(env)

        for el in L:
            if el == v:
                return True
        return False
    def __repr__(self):
        return f"({repr(self.element)} ∈ {repr(self.list)})"

    def __eq__(self, other):

        return isinstance(other, In) and self.element == other.element and self.list == other.list



class Exists:
    def __init__(self, function, list):
        self.function = function

        self.list = list

    def __repr__(self):
        param = self.function.params[0]
        return f"(∃ {param} ∈ {repr(self.list)}, {repr(self.function)})"

    def __eq__(self, other):
        return isinstance(other, Exists) and self.function == other.function and self.list == other.list

        
    def execute(self, env):
        #print([e.body for e in env])
        L = self.list.execute(env)
        for element in L:
            T = Apply(self.function, Color(element)).execute(env)
            if T:
                return True
        return False





class Forall:
    def __init__(self, function, list):
        self.function = function
        self.list = list

    def execute(self, env):

        L = self.list.execute(env)
        for element in L:
            if not Apply(self.function, Color(element)).execute(env):
                return False
        return True

    def __repr__(self):
        param = self.function.params[0]
        return f"(∀ {param} ∈ {repr(self.list)}, {repr(self.function)})"

    def __eq__(self, other):
        return isinstance(other, Forall) and self.function == other.function and self.list == other.list



class ExistsUnique:
    def __init__(self, function, lst):
        self.function = function
        self.list = lst

    def __repr__(self):
        param = self.function.params[0]
        return f"(∃! {param} ∈ {repr(self.list)}, {repr(self.function)})"

    def __eq__(self, other):
        return isinstance(other, ExistsUnique) and self.function == other.function and self.list == other.list

    def execute(self, env):
        L = self.list.execute(env)
        true_count = 0
        for element in L:
            T = Apply(self.function, Color(element)).execute(env)
            if T:
                true_count += 1
                if true_count > 1:
                    return False
        return true_count == 1




import unittest

class TestPrimitives(unittest.TestCase):
    def test_number(self):
        num = Number(5)
        self.assertEqual(num.execute({}), 5)

    def test_plus(self):
        expr = Plus(Number(3), Number(4))
        self.assertEqual(expr.execute({}), 7)

    def test_list(self):
        lst = List(Number(1), Number(2), Number(3))
        self.assertEqual(lst.execute({}), [1, 2, 3])

    def test_map(self):
        lst = List(Number(1), Number(2), Number(3))
        func = Lambda("x", Plus(Var("x"), Number(1)))
        mapped = Map(func, lst)
        self.assertEqual(mapped.execute({}), [2, 3, 4])

    def test_exists(self):
        lst = List(Number(1), Number(2), Number(3))
        func = Lambda("x", Equals(Var("x"), Number(2)))
        exists = Exists(func, lst)
        self.assertTrue(exists.execute({}))

    def test_forall(self):
        lst = List(Number(1), Number(2), Number(3))
        func = Lambda("x", Or(Equals(Var("x"), Number(2)), Equals(Var("x"), Number(3))))
        forall = Forall(func, lst)
        self.assertFalse(forall.execute({}))

    def test_index(self):
        lst = List(Number(1), Number(2), Number(3))
        idx = Index(lst, Number(1))
        self.assertEqual(idx.execute({}), 2)

        idx_out_of_bounds = Index(lst, Number(3))
        self.assertEqual(idx_out_of_bounds.execute({}), Number(None))

    def test_subset(self):
        lst = List(Number(1), Number(2), Number(3), Number(4), Number(5))
        subset = Subset(lst, Number(1), Number(4), Number(2))
        self.assertEqual(subset.execute({}), [2,4])

        # Test invalid bounds and increment
        invalid_subset = Subset(lst, Number(4), Number(1), Number(1))
        self.assertEqual(invalid_subset.execute({}), [])

        invalid_increment = Subset(lst, Number(1), Number(4), Number(0))
        self.assertEqual(invalid_increment.execute({}), [])


if __name__ == '__main__':

    #unittest.main()
    C = List(Number(0), Number(2))
    G = List(Number(0), Number(1), Number(2), Number(3))

    L = Lambda(["x0", "x1"], In(Index(Var("x0"), Number(1)), Var("x1")))

    expr = Statement(L)


    print(expr.evaluate([C, G]))

