"""PyGo Autograd Engine — reverse-mode automatic differentiation."""
from __future__ import annotations
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Dict, Tuple, Any
import weakref


# ─── Grad Mode ───────────────────────────────────────────────────────────────

_grad_mode = threading.local()

def is_grad_enabled() -> bool:
    return getattr(_grad_mode, "enabled", True)

def set_grad_enabled(value: bool):
    _grad_mode.enabled = value

class no_grad:
    def __enter__(self):
        self._prev = is_grad_enabled()
        set_grad_enabled(False)
    def __exit__(self, *_):
        set_grad_enabled(self._prev)
    def __call__(self, fn):
        import functools
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with no_grad():
                return fn(*args, **kwargs)
        return wrapper

class enable_grad:
    def __enter__(self):
        self._prev = is_grad_enabled()
        set_grad_enabled(True)
    def __exit__(self, *_):
        set_grad_enabled(self._prev)

class set_grad_enabled_ctx:
    def __init__(self, mode: bool):
        self.mode = mode
    def __enter__(self):
        self._prev = is_grad_enabled()
        set_grad_enabled(self.mode)
    def __exit__(self, *_):
        set_grad_enabled(self._prev)


# ─── Node ────────────────────────────────────────────────────────────────────

@dataclass
class Edge:
    fn:       "Node"
    input_nr: int

@dataclass
class Node:
    """A node in the autograd computation graph."""
    fn:            Callable              # backward function
    name:          str       = ""
    next_edges:    List[Edge]= field(default_factory=list)
    saved_tensors: List[Any] = field(default_factory=list)
    output_nr:     int       = 0
    is_leaf:       bool      = False
    _seq_nr:       int       = 0        # topological sort ordering

    _seq_counter = 0
    _seq_lock    = threading.Lock()

    def __post_init__(self):
        with Node._seq_lock:
            Node._seq_counter += 1
            self._seq_nr = Node._seq_counter

    def save_for_backward(self, *tensors):
        self.saved_tensors = list(tensors)

    def apply(self, grads: list) -> list:
        return self.fn(*grads)

    def __repr__(self):
        return f"<Node {self.name} seq={self._seq_nr}>"


# ─── Accumulate Grad Node ─────────────────────────────────────────────────────

class AccumulateGrad(Node):
    """Leaf node: accumulates gradients into a variable."""
    def __init__(self, variable):
        super().__init__(fn=self._apply, name="AccumulateGrad", is_leaf=True)
        self.variable = weakref.ref(variable)

    def _apply(self, *grads):
        var = self.variable()
        if var is None:
            return []
        g = grads[0] if grads else None
        if g is None:
            return []
        if var._grad is None:
            var._grad = g.clone()
        else:
            var._grad += g
        return []


# ─── Engine ──────────────────────────────────────────────────────────────────

class Engine:
    """Reverse-mode autodiff engine using Kahn's topological sort + BFS."""

    _local = threading.local()

    @classmethod
    def backward(
        cls,
        roots:          List[Any],
        root_grads:     List[Optional[Any]],
        retain_graph:   bool = False,
        create_graph:   bool = False,
        inputs:         Optional[List[Any]] = None,
    ) -> None:
        """
        Run backward from `roots` with initial gradients `root_grads`.
        roots      — list of Tensor (each with .grad_fn)
        root_grads — matching initial grad tensors (or None → ones_like)
        """
        # Collect all nodes via BFS, build in-degree map
        topo_order, in_degree = cls._build_topo(roots)

        # Initialise grad map: node → list of accumulated grads (one per output)
        grad_map: Dict[Node, List[Optional[Any]]] = defaultdict(list)

        for root, root_grad in zip(roots, root_grads):
            fn = root.grad_fn
            if fn is None:
                # Leaf — accumulate directly
                if root.requires_grad:
                    if root._grad is None:
                        import copy
                        root._grad = copy.copy(root_grad) if root_grad is not None \
                                     else cls._ones_like(root)
                    else:
                        root._grad = root._grad + (root_grad if root_grad is not None
                                                   else cls._ones_like(root))
                continue

            g = root_grad if root_grad is not None else cls._ones_like(root)
            grad_map[fn].append(g)

        # Process nodes in reverse topological order
        for node in topo_order:
            if node not in grad_map:
                continue

            grads_in = grad_map[node]
            # Sum multiple grad contributions (e.g. node has multiple outputs)
            combined = cls._reduce_grads(grads_in)

            try:
                outputs = node.apply(combined)
            except Exception as e:
                raise RuntimeError(f"Error in backward of {node.name}: {e}") from e

            if not retain_graph:
                node.saved_tensors = []

            # Distribute to next nodes
            for i, edge in enumerate(node.next_edges):
                if edge is None or edge.fn is None:
                    continue
                g_out = outputs[i] if i < len(outputs) else None
                if g_out is None:
                    continue
                grad_map[edge.fn].append(g_out)

    @staticmethod
    def _build_topo(roots: list) -> Tuple[list, dict]:
        """Topological sort (reverse BFS from roots)."""
        visited   = set()
        in_degree : Dict[Node, int] = defaultdict(int)
        all_nodes = []

        queue = []
        for r in roots:
            if hasattr(r, "grad_fn") and r.grad_fn is not None:
                queue.append(r.grad_fn)

        while queue:
            node = queue.pop()
            if id(node) in visited:
                continue
            visited.add(id(node))
            all_nodes.append(node)
            for edge in node.next_edges:
                if edge and edge.fn:
                    in_degree[edge.fn] += 1
                    queue.append(edge.fn)

        # Sort by seq_nr descending (later ops first = reverse topo)
        all_nodes.sort(key=lambda n: -n._seq_nr)
        return all_nodes, in_degree

    @staticmethod
    def _reduce_grads(grads: list):
        """Sum a list of gradient tensors."""
        result = None
        for g in grads:
            if g is None:
                continue
            result = g if result is None else result + g
        return result

    @staticmethod
    def _ones_like(t):
        try:
            import numpy as np
            return np.ones_like(t.data) if hasattr(t, "data") else 1.0
        except Exception:
            return 1.0


# ─── Custom Function API ─────────────────────────────────────────────────────

class Function:
    """
    Base class for custom differentiable functions.

    Usage:
        class MySquare(Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x ** 2

            @staticmethod
            def backward(ctx, grad_out):
                x, = ctx.saved_tensors
                return 2 * x * grad_out

        y = MySquare.apply(x)
    """

    @staticmethod
    def forward(ctx, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError

    @classmethod
    def apply(cls, *args):
        from ..tensor import Tensor

        # Collect inputs that need grad
        needs_grad = any(
            isinstance(a, Tensor) and a.requires_grad
            for a in args
        )

        ctx = _FunctionCtx()
        outputs = cls.forward(ctx, *args)

        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        if is_grad_enabled() and needs_grad:
            for out in outputs:
                if not isinstance(out, Tensor):
                    continue

                def make_bwd(ctx_ref, cls_ref):
                    def bwd(*grad_outs):
                        grads = cls_ref.backward(ctx_ref, *grad_outs)
                        if not isinstance(grads, tuple):
                            grads = (grads,)
                        return list(grads)
                    return bwd

                node = Node(
                    fn   = make_bwd(ctx, cls),
                    name = cls.__name__,
                )
                node.next_edges = [
                    Edge(fn=a.grad_fn, input_nr=0)
                    for a in args
                    if isinstance(a, Tensor) and a.grad_fn is not None
                ]
                node.saved_tensors = ctx._saved
                out.grad_fn = node
                out.requires_grad = True

        return outputs[0] if len(outputs) == 1 else outputs


class _FunctionCtx:
    def __init__(self):
        self._saved         = []
        self.needs_input_grad: List[bool] = []

    def save_for_backward(self, *tensors):
        self._saved = list(tensors)

    @property
    def saved_tensors(self):
        return tuple(self._saved)


# ─── Grad utilities ──────────────────────────────────────────────────────────

def grad(outputs, inputs, grad_outputs=None, retain_graph=False,
         create_graph=False, only_inputs=True) -> List:
    """Compute gradients of `outputs` w.r.t. `inputs`."""
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]
    if grad_outputs is None:
        grad_outputs = [None] * len(outputs)

    Engine.backward(outputs, grad_outputs,
                    retain_graph=retain_graph,
                    create_graph=create_graph,
                    inputs=inputs)

    return [inp._grad for inp in inputs]


def jacobian(f, inputs, create_graph=False):
    """Compute full Jacobian matrix via repeated backward."""
    from ..tensor import Tensor
    import numpy as np

    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    outputs = f(*inputs)
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]

    out_size = sum(o.numel() for o in outputs)
    in_size  = sum(i.numel() for i in inputs)

    J = np.zeros((out_size, in_size))
    for oi in range(out_size):
        # zero grads
        for inp in inputs:
            if inp._grad is not None:
                inp._grad = None
        # unit vector
        e = [Tensor.zeros_like(o) for o in outputs]
        row, col = divmod(oi, max(o.numel() for o in outputs))
        e[row].data.flat[col] = 1.0

        Engine.backward(outputs, e, retain_graph=True, create_graph=create_graph)
        col_offset = 0
        for inp in inputs:
            g = inp._grad
            if g is not None:
                J[oi, col_offset:col_offset+inp.numel()] = g.data.flatten()
            col_offset += inp.numel()

    return J


def hessian(f, inputs):
    """Compute Hessian via double-backward."""
    def grad_f(*args):
        outputs = f(*args)
        g = grad(outputs, list(args), create_graph=True)
        from ..tensor import Tensor
        return Tensor.stack(g).flatten()
    return jacobian(grad_f, inputs, create_graph=True)


# ─── Activation Checkpoint ───────────────────────────────────────────────────

class CheckpointFunction(Function):
    """Re-runs forward during backward to trade compute for memory."""

    @staticmethod
    def forward(ctx, run_fn, *args):
        ctx.run_fn = run_fn
        ctx.save_for_backward(*args)
        with no_grad():
            outputs = run_fn(*args)
        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        inputs = ctx.saved_tensors
        with enable_grad():
            outputs = ctx.run_fn(*inputs)

        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        Engine.backward(list(outputs), list(grad_outputs), retain_graph=False)
        return (None,) + tuple(inp._grad for inp in inputs)


def checkpoint(fn, *args, use_reentrant=True):
    """Memory-efficient forward using activation checkpointing."""
    return CheckpointFunction.apply(fn, *args)
