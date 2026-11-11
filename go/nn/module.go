// Package nn provides neural network building blocks for PyGo.
// Architecture: Go Module API → CGo → Python nn.Module → C++ kernels
package nn

import (
	"encoding/json"
	"fmt"
	"github.com/pygo/go/tensor"
	"math"
	"os"
	"sync"
)

// ─── Parameter ───────────────────────────────────────────────────────────────

// Parameter is a tensor that participates in gradient computation.
type Parameter struct {
	*tensor.Tensor
	name string
}

func NewParameter(t *tensor.Tensor) *Parameter {
	t.SetRequiresGrad(true)
	return &Parameter{Tensor: t}
}

// ─── Module ──────────────────────────────────────────────────────────────────

// Module is the base interface for all PyGo neural network components.
type Module interface {
	Forward(inputs ...*tensor.Tensor) *tensor.Tensor
	Parameters() []*Parameter
	NamedParameters() map[string]*Parameter
	Modules() []Module
	NamedModules() map[string]Module
	Train(mode bool)
	IsTraining() bool
	To(dev tensor.Device) Module
	ZeroGrad()
	StateDict() map[string]*tensor.Tensor
	LoadStateDict(sd map[string]*tensor.Tensor) error
	String() string
}

// ─── BaseModule ──────────────────────────────────────────────────────────────

// BaseModule provides default implementations for Module methods.
// Embed this in every custom module (like PyTorch's nn.Module).
type BaseModule struct {
	mu       sync.RWMutex
	params   map[string]*Parameter
	buffers  map[string]*tensor.Tensor
	children map[string]Module
	training bool
	name     string
}

func NewBase(name string) BaseModule {
	return BaseModule{
		params:   make(map[string]*Parameter),
		buffers:  make(map[string]*tensor.Tensor),
		children: make(map[string]Module),
		training: true,
		name:     name,
	}
}

func (m *BaseModule) RegisterParameter(name string, p *Parameter) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.params[name] = p
}

func (m *BaseModule) RegisterBuffer(name string, t *tensor.Tensor) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.buffers[name] = t
}

func (m *BaseModule) RegisterModule(name string, child Module) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.children[name] = child
}

func (m *BaseModule) Parameters() []*Parameter {
	m.mu.RLock()
	defer m.mu.RUnlock()
	var ps []*Parameter
	for _, p := range m.params {
		ps = append(ps, p)
	}
	for _, child := range m.children {
		ps = append(ps, child.Parameters()...)
	}
	return ps
}

func (m *BaseModule) NamedParameters() map[string]*Parameter {
	m.mu.RLock()
	defer m.mu.RUnlock()
	out := make(map[string]*Parameter)
	for k, v := range m.params {
		out[k] = v
	}
	for childName, child := range m.children {
		for k, v := range child.NamedParameters() {
			out[childName+"."+k] = v
		}
	}
	return out
}

func (m *BaseModule) Modules() []Module {
	m.mu.RLock()
	defer m.mu.RUnlock()
	var mods []Module
	for _, c := range m.children {
		mods = append(mods, c)
		mods = append(mods, c.Modules()...)
	}
	return mods
}

func (m *BaseModule) NamedModules() map[string]Module {
	m.mu.RLock()
	defer m.mu.RUnlock()
	out := make(map[string]Module)
	for name, child := range m.children {
		out[name] = child
		for k, v := range child.NamedModules() {
			out[name+"."+k] = v
		}
	}
	return out
}

func (m *BaseModule) Train(mode bool) {
	m.mu.Lock()
	m.training = mode
	m.mu.Unlock()
	for _, child := range m.children {
		child.Train(mode)
	}
}

func (m *BaseModule) IsTraining() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.training
}

func (m *BaseModule) ZeroGrad() {
	for _, p := range m.Parameters() {
		p.ZeroGrad()
	}
}

func (m *BaseModule) To(dev tensor.Device) Module {
	for _, p := range m.params {
		*p = *NewParameter(p.To(dev))
	}
	for k, b := range m.buffers {
		m.buffers[k] = b.To(dev)
	}
	for _, child := range m.children {
		child.To(dev)
	}
	return nil // caller wraps
}

func (m *BaseModule) StateDict() map[string]*tensor.Tensor {
	out := make(map[string]*tensor.Tensor)
	for k, p := range m.NamedParameters() {
		out[k] = p.Tensor
	}
	for k, b := range m.buffers {
		out[k] = b
	}
	return out
}

func (m *BaseModule) LoadStateDict(sd map[string]*tensor.Tensor) error {
	np := m.NamedParameters()
	for k, t := range sd {
		if p, ok := np[k]; ok {
			if !p.Shape().Equals(t.Shape()) {
				return fmt.Errorf("shape mismatch for %q: got %v want %v",
					k, t.Shape(), p.Shape())
			}
			*p.Tensor = *t
		}
	}
	return nil
}

func (m *BaseModule) String() string {
	return m.name
}

// ─── Sequential ──────────────────────────────────────────────────────────────

// Sequential runs modules in order.
type Sequential struct {
	BaseModule
	layers []Module
}

func NewSequential(layers ...Module) *Sequential {
	s := &Sequential{BaseModule: NewBase("Sequential")}
	for i, l := range layers {
		s.layers = append(s.layers, l)
		s.RegisterModule(fmt.Sprintf("%d", i), l)
	}
	return s
}

func (s *Sequential) Append(m Module) *Sequential {
	s.layers = append(s.layers, m)
	s.RegisterModule(fmt.Sprintf("%d", len(s.layers)-1), m)
	return s
}

func (s *Sequential) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	x := inputs[0]
	for _, l := range s.layers {
		x = l.Forward(x)
	}
	return x
}

func (s *Sequential) String() string {
	out := "Sequential(\n"
	for i, l := range s.layers {
		out += fmt.Sprintf("  (%d): %s\n", i, l.String())
	}
	return out + ")"
}

// ─── Linear ──────────────────────────────────────────────────────────────────

// Linear applies y = x @ W.T + b
type Linear struct {
	BaseModule
	InFeatures  int
	OutFeatures int
	UseBias     bool
	Weight      *Parameter // [out, in]
	Bias        *Parameter // [out]
}

func NewLinear(in, out int, bias bool) *Linear {
	l := &Linear{
		BaseModule:  NewBase(fmt.Sprintf("Linear(in=%d, out=%d)", in, out)),
		InFeatures:  in,
		OutFeatures: out,
		UseBias:     bias,
	}
	// Kaiming uniform init (fan_in mode)
	std := math.Sqrt(2.0 / float64(in))
	w := tensor.Randn(tensor.Shape{int64(out), int64(in)})
	w = w.MulScalar(std)
	l.Weight = NewParameter(w)
	l.RegisterParameter("weight", l.Weight)

	if bias {
		b := tensor.Zeros(tensor.Shape{int64(out)})
		l.Bias = NewParameter(b)
		l.RegisterParameter("bias", l.Bias)
	}
	return l
}

func (l *Linear) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	x := inputs[0]
	// x @ W.T  →  matmul with weight transposed
	wT := l.Weight.Transpose(-2, -1)
	out := x.MatMul(wT)
	if l.UseBias {
		out = out.Add(l.Bias.Tensor)
	}
	return out
}

func (l *Linear) String() string {
	return fmt.Sprintf("Linear(in=%d, out=%d, bias=%v)", l.InFeatures, l.OutFeatures, l.UseBias)
}

// ─── LayerNorm ───────────────────────────────────────────────────────────────

type LayerNorm struct {
	BaseModule
	NormalizedShape []int
	Eps             float64
	Weight          *Parameter
	Bias            *Parameter
}

func NewLayerNorm(normalizedShape []int, eps float64) *LayerNorm {
	ln := &LayerNorm{
		BaseModule:      NewBase("LayerNorm"),
		NormalizedShape: normalizedShape,
		Eps:             eps,
	}
	dim := int64(normalizedShape[len(normalizedShape)-1])
	ln.Weight = NewParameter(tensor.Ones(tensor.Shape{dim}))
	ln.Bias = NewParameter(tensor.Zeros(tensor.Shape{dim}))
	ln.RegisterParameter("weight", ln.Weight)
	ln.RegisterParameter("bias", ln.Bias)
	return ln
}

func (ln *LayerNorm) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	x := inputs[0]
	dim := int64(ln.NormalizedShape[len(ln.NormalizedShape)-1])
	mean := x.Mean(-1, true)
	xc := x.Add(mean.MulScalar(-1))
	v := xc.Pow(2).Mean(-1, true).AddScalar(ln.Eps)
	std := v.Sqrt()
	xn := xc.Div(std)
	// scale & shift
	_ = dim
	out := xn.Mul(ln.Weight.Tensor).Add(ln.Bias.Tensor)
	return out
}

// ─── Embedding ───────────────────────────────────────────────────────────────

type Embedding struct {
	BaseModule
	NumEmbeddings int
	EmbeddingDim  int
	PaddingIdx    int
	Weight        *Parameter // [vocab, dim]
}

func NewEmbedding(numEmbeddings, embeddingDim int, padding ...int) *Embedding {
	paddingIdx := -1
	if len(padding) > 0 {
		paddingIdx = padding[0]
	}
	e := &Embedding{
		BaseModule:    NewBase(fmt.Sprintf("Embedding(%d, %d)", numEmbeddings, embeddingDim)),
		NumEmbeddings: numEmbeddings,
		EmbeddingDim:  embeddingDim,
		PaddingIdx:    paddingIdx,
	}
	// Normal init std=1
	w := tensor.Randn(tensor.Shape{int64(numEmbeddings), int64(embeddingDim)})
	e.Weight = NewParameter(w)
	e.RegisterParameter("weight", e.Weight)
	return e
}

func (e *Embedding) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	if len(inputs) == 0 {
		panic("nn.Embedding: missing index tensor")
	}
	panic("nn.Embedding: native embedding lookup is not exposed by the stable Go tensor ABI")
}

// ─── Dropout ─────────────────────────────────────────────────────────────────

type Dropout struct {
	BaseModule
	P float64
}

func NewDropout(p float64) *Dropout {
	return &Dropout{BaseModule: NewBase(fmt.Sprintf("Dropout(p=%.2f)", p)), P: p}
}

func (d *Dropout) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	x := inputs[0]
	if !d.IsTraining() || d.P == 0 {
		return x
	}
	panic("nn.Dropout: native dropout kernel is not exposed by the stable Go tensor ABI")
}

// ─── ModuleList ───────────────────────────────────────────────────────────────

type ModuleList struct {
	BaseModule
	mods []Module
}

func NewModuleList(mods ...Module) *ModuleList {
	ml := &ModuleList{BaseModule: NewBase("ModuleList")}
	for i, m := range mods {
		ml.mods = append(ml.mods, m)
		ml.RegisterModule(fmt.Sprintf("%d", i), m)
	}
	return ml
}

func (ml *ModuleList) Get(i int) Module { return ml.mods[i] }
func (ml *ModuleList) Len() int         { return len(ml.mods) }
func (ml *ModuleList) Append(m Module)  { ml.mods = append(ml.mods, m) }
func (ml *ModuleList) Forward(inputs ...*tensor.Tensor) *tensor.Tensor {
	panic("ModuleList has no Forward; iterate with Get()")
}

// ─── Freeze / Unfreeze ────────────────────────────────────────────────────────

func Freeze(m Module) {
	for _, p := range m.Parameters() {
		p.SetRequiresGrad(false)
	}
}

func Unfreeze(m Module) {
	for _, p := range m.Parameters() {
		p.SetRequiresGrad(true)
	}
}

// ─── Parameter Count ─────────────────────────────────────────────────────────

func CountParams(m Module) (total int64, trainable int64) {
	seen := make(map[*Parameter]bool)
	for _, p := range m.Parameters() {
		if seen[p] {
			continue
		}
		seen[p] = true
		n := p.NumEl()
		total += n
		if p.RequiresGrad() {
			trainable += n
		}
	}
	return
}

// ─── JSON State Dict ─────────────────────────────────────────────────────────

func SaveStateDict(m Module, path string) error {
	sd := m.StateDict()
	meta := make(map[string]interface{})
	for k, t := range sd {
		meta[k] = map[string]interface{}{
			"shape": t.Shape(),
			"dtype": t.DType().String(),
		}
	}
	data, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		return err
	}
	return nil
}

// LoadStateDictFile loads the metadata state-dict format produced by SaveStateDict.
// The metadata format is intentionally shape/dtype-only; use SafeTensors for portable tensor bytes.
func LoadStateDictFile(m Module, path string) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var meta map[string]struct {
		Shape []int64 `json:"shape"`
		DType string  `json:"dtype"`
	}
	if err := json.Unmarshal(raw, &meta); err != nil {
		return err
	}
	current := m.StateDict()
	for name, info := range meta {
		t, ok := current[name]
		if !ok {
			continue
		}
		if !t.Shape().Equals(tensor.Shape(info.Shape)) {
			return fmt.Errorf("shape mismatch for %q", name)
		}
	}
	return nil
}

// ─── Apply ────────────────────────────────────────────────────────────────────

func Apply(m Module, fn func(Module)) {
	fn(m)
	for _, child := range m.Modules() {
		fn(child)
	}
}
