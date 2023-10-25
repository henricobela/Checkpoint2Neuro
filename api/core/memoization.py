memo = {}

def memoize(func):
    def wrapper(*args):
        if args in memo:
            return memo[args]
        result = func(*args)
        memo[args] = result
        return result
    return wrapper