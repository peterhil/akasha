#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Monad in Python (with nice syntax!)
# http://www.valuedlessons.com/2008/01/monads-in-python-with-nice-syntax.html

import types


class Monad:
    def bind(self, func):
        raise NotImplementedError

    def __rshift__(self, bindee):
        return self.bind(bindee)

    def __add__(self, bindee_without_arg):
        return self.bind(lambda _: bindee_without_arg())


def make_decorator(func, *dec_args):
    def decorator(undecorated):
        def decorated(*args, **kargs):
            return func(undecorated, args, kargs, *dec_args)

        decorated.__name__ = undecorated.__name__
        return decorated

    decorator.__name__ = func.__name__
    return decorator


def make_decorator_with_args(func):
    def decorator_with_args(*dec_args):
        return make_decorator(func, *dec_args)
    return decorator_with_args


decorator = make_decorator
decorator_with_args = make_decorator_with_args


@decorator_with_args
def do(func, func_args, func_kargs, Monad):
    @handle_monadic_throws(Monad)
    def run_maybe_iterator():
        itr = func(*func_args, **func_kargs)

        if isinstance(itr, types.GeneratorType):
            @handle_monadic_throws(Monad)
            def send(val):
                try:
                    # here's the real magic
                    monad = itr.send(val)
                    return monad.bind(send)
                except StopIteration:
                    return Monad.unit(None)

            return send(None)
        else:
            #not really a generator
            if itr is None:
                return Monad.unit(None)
            else:
                return itr

    return run_maybe_iterator()


@decorator_with_args
def handle_monadic_throws(func, func_args, func_kargs, Monad):
    try:
        return func(*func_args, **func_kargs)
    except MonadReturn as ret:
        return Monad.unit(ret.value)
    except Done as done:
        assert isinstance(done.monad, Monad)
        return done.monad


class MonadReturn(Exception):
    def __init__(self, value):
        self.value = value
        Exception.__init__(self, value)


class Done(Exception):
    def __init__(self, monad):
        self.monad = monad
        Exception.__init__(self, monad)


def mreturn(val):
    raise MonadReturn(val)


def done(val):
    raise Done(val)


def fid(val):
    return val


class Failable(Monad):
    def __init__(self, value, success):
        self.value = value
        self.success = success

    def __repr__(self):
        if self.success:
            return "Success(%r)" % (self.value,)
        else:
            return "Failure(%r)" % (self.value,)

    def bind(self, bindee):
        if self.success:
            return bindee(self.value)
        else:
            return self

    @classmethod
    def unit(cls, val):
        return cls(val, True)


class Success(Failable):
    def __init__(self, value):
        Failable.__init__(self, value, True)


class Failure(Failable):
    def __init__(self, value):
        Failable.__init__(self, value, False)


def failable_monad_examle():
    def fdiv(a, b):
        if b == 0:
            return Failure("cannot divide by zero")
        else:
            return Success(a / b)

    @do(Failable)
    def with_failable(first_divisor):
        val1 = yield fdiv(2.0, first_divisor)
        val2 = yield fdiv(3.0, 1.0)
        val3 = yield fdiv(val1, val2)
        mreturn(val3)

    print(with_failable(0.0))
    print(with_failable(1.0))


class StateChanger(Monad):
    def __init__(self, run):
        self.run = run

    def bind(self, bindee):
        run0 = self.run

        def run1(state0):
            (result, state1) = run0(state0)
            return bindee(result).run(state1)

        return StateChanger(run1)

    @classmethod
    def unit(cls, val):
        return cls(lambda state: (val, state))


def get_state(view=fid):
    return change_state(fid, view)


def change_state(changer, view=fid):
    def make_new_state(old_state):
        new_state = changer(old_state)
        viewed_state = view(old_state)
        return (viewed_state, new_state)
    return StateChanger(make_new_state)


def state_changer_monad_example():
    @do(StateChanger)
    def dict_state_copy(key1, key2):
        val = yield dict_state_get(key1)
        yield dict_state_set(key2, val)
        mreturn(val)

    @do(StateChanger)
    def dict_state_get(key, default=None):
        dct = yield get_state()
        val = dct.get(key, default)
        mreturn(val)

    @do(StateChanger)
    def dict_state_set(key, val):
        def dict_set(dct, key, val):
            dct[key] = val
            return dct

        new_state = yield change_state(lambda dct: dict_set(dct, key, val))
        mreturn(val)

    @do(StateChanger)
    def with_dict_state():
        val2 = yield dict_state_set("a", 2)
        yield dict_state_copy("a", "b")
        state = yield get_state()
        mreturn(val2)

    print(with_dict_state().run({}))  # (2, {"a" : 2, "b" : 2})


class ContinuationMonad(Monad):
    def __init__(self, run):
        self.run = run

    def __call__(self, cont=fid):
        return self.run(cont)

    def bind(self, bindee):
        return ContinuationMonad(lambda cont: self.run(lambda val: bindee(val).run(cont)))

    @classmethod
    def unit(cls, val):
        return cls(lambda cont: cont(val))

    @classmethod
    def zero(cls):
        return cls(lambda cont: None)


def callcc(usecc):
    return ContinuationMonad(
        lambda cont: usecc(
            lambda val: ContinuationMonad(
                lambda _: cont(val))).run(cont))


def continuation_monad_example():
    from collections import deque

    class Mailbox:
        def __init__(self):
            self.messages = deque()
            self.handlers = deque()

        def send(self, message):
            if self.handlers:
                handler = self.handlers.popleft()
                handler(message)()
            else:
                self.messages.append(message)

        def receive(self):
            return callcc(self.react)

        @do(ContinuationMonad)
        def react(self, handler):
            if self.messages:
                message = self.messages.popleft()
                yield handler(message)
            else:
                self.handlers.append(handler)
                done(ContinuationMonad.zero())

    @do(ContinuationMonad)
    def insert(mb, values):
        for val in values:
            mb.send(val)

    @do(ContinuationMonad)
    def multiply(mbin, mbout, factor):
        while True:
            val = (yield mbin.receive())
            mbout.send(val * factor)

    @do(ContinuationMonad)
    def print_all(mb):
        while True:
            print((yield mb.receive()))

    original = Mailbox()
    multiplied = Mailbox()

    print_all(multiplied)()
    multiply(original, multiplied, 2)()
    insert(original, [1, 2, 3])()


if __name__ == '__main__':
    failable_monad_examle()
    state_changer_monad_example()
    continuation_monad_example()
