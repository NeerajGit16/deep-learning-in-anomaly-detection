"""Stand-in for the fastodeint C++ extension, which PenSimPy imports at the top; adlab.pensim never calls it."""


def integrate(*args, **kwargs):
    raise NotImplementedError("fastodeint (C++) is not installed; adlab.pensim runs PenSimPy with fast=False")
