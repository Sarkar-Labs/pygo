// Package training provides a high-level training loop for PyGo models.
package training

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/pygo/go/callbacks"
	"github.com/pygo/go/data"
	"github.com/pygo/go/metrics"
	"github.com/pygo/go/nn"
	"github.com/pygo/go/optim"
	"github.com/pygo/go/serialization"
	"github.com/pygo/go/tensor"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// ─── TrainingConfig ──────────────────────────────────────────────────────────

type TrainingConfig struct {
	MaxEpochs       int
	MaxSteps        int     // overrides MaxEpochs if > 0
	GradClipNorm    float64 // 0 = disabled
	GradAccumSteps  int     // gradient accumulation
	MixedPrecision  bool    // AMP (auto mixed precision)
	LogEvery        int     // steps
	EvalEvery       int     // steps
	SaveEvery       int     // steps
	CheckpointDir   string
	ResumeFrom      string
	DetectAnomalies bool // NaN / Inf detection
	NumWorkers      int
	PinMemory       bool
}

func DefaultConfig() TrainingConfig {
	return TrainingConfig{
		MaxEpochs:      10,
		GradClipNorm:   1.0,
		GradAccumSteps: 1,
		LogEvery:       10,
		EvalEvery:      100,
		SaveEvery:      500,
		NumWorkers:     4,
	}
}

// ─── TrainingState ───────────────────────────────────────────────────────────

type TrainingState struct {
	Epoch       int
	GlobalStep  int
	BestMetric  float64
	BestStep    int
	TrainLosses []float64
	ValLosses   []float64
	LRHistory   []float64
	ElapsedTime time.Duration
	mu          sync.RWMutex
}

func (s *TrainingState) Update(loss float64, lr float64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.GlobalStep++
	s.TrainLosses = append(s.TrainLosses, loss)
	s.LRHistory = append(s.LRHistory, lr)
}

func (s *TrainingState) LastNLossMean(n int) float64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if len(s.TrainLosses) == 0 {
		return math.NaN()
	}
	start := max(0, len(s.TrainLosses)-n)
	sum := 0.0
	for _, l := range s.TrainLosses[start:] {
		sum += l
	}
	return sum / float64(len(s.TrainLosses[start:]))
}

// ─── Trainer ─────────────────────────────────────────────────────────────────

type Trainer struct {
	model     nn.Module
	optimizer optim.Optimizer
	scheduler optim.Scheduler
	lossFn    LossFn
	cfg       TrainingConfig
	state     TrainingState
	callbacks []callbacks.Callback
	metricFns []metrics.MetricFn
	device    tensor.Device
	ctx       context.Context
	cancelFn  context.CancelFunc
}

// LossFn computes the scalar loss from model output and targets.
type LossFn func(output, target *tensor.Tensor) *tensor.Tensor

// ─── Constructor ─────────────────────────────────────────────────────────────

func NewTrainer(
	model nn.Module,
	optimizer optim.Optimizer,
	lossFn LossFn,
	cfg TrainingConfig,
	opts ...TrainerOption,
) *Trainer {
	ctx, cancel := context.WithCancel(context.Background())
	t := &Trainer{
		model:     model,
		optimizer: optimizer,
		lossFn:    lossFn,
		cfg:       cfg,
		device:    tensor.DevCPU,
		ctx:       ctx,
		cancelFn:  cancel,
	}
	for _, o := range opts {
		o(t)
	}
	return t
}

// ─── Options ─────────────────────────────────────────────────────────────────

type TrainerOption func(*Trainer)

func WithScheduler(s optim.Scheduler) TrainerOption { return func(t *Trainer) { t.scheduler = s } }
func WithCallbacks(cs ...callbacks.Callback) TrainerOption {
	return func(t *Trainer) { t.callbacks = append(t.callbacks, cs...) }
}
func WithMetrics(ms ...metrics.MetricFn) TrainerOption {
	return func(t *Trainer) { t.metricFns = append(t.metricFns, ms...) }
}
func WithDevice(d tensor.Device) TrainerOption { return func(t *Trainer) { t.device = d } }

// ─── Fit ─────────────────────────────────────────────────────────────────────

func (t *Trainer) Fit(
	trainLoader data.DataLoader,
	valLoader data.DataLoader,
) error {
	// Resume if checkpoint exists
	if t.cfg.ResumeFrom != "" {
		if err := t.loadCheckpoint(t.cfg.ResumeFrom); err != nil {
			return fmt.Errorf("resume: %w", err)
		}
	}

	t.fireEvent(callbacks.EventTrainBegin, nil)
	startTime := time.Now()

	for epoch := t.state.Epoch; epoch < t.cfg.MaxEpochs; epoch++ {
		select {
		case <-t.ctx.Done():
			return context.Canceled
		default:
		}

		t.state.Epoch = epoch
		t.fireEvent(callbacks.EventEpochBegin, map[string]interface{}{"epoch": epoch})

		// ── Train epoch ──────────────────────────────────────────────────
		trainLoss, err := t.trainEpoch(trainLoader)
		if err != nil {
			return fmt.Errorf("epoch %d train: %w", epoch, err)
		}

		// ── Validation ───────────────────────────────────────────────────
		var valLoss float64
		if valLoader != nil {
			valLoss, err = t.evalEpoch(valLoader)
			if err != nil {
				return fmt.Errorf("epoch %d val: %w", epoch, err)
			}
			t.state.ValLosses = append(t.state.ValLosses, valLoss)

			// Track best
			if valLoss < t.state.BestMetric || t.state.BestMetric == 0 {
				t.state.BestMetric = valLoss
				t.state.BestStep = t.state.GlobalStep
				if t.cfg.CheckpointDir != "" {
					_ = t.saveCheckpoint(t.cfg.CheckpointDir + "/best.pt")
				}
			}
		}

		epochMeta := map[string]interface{}{
			"epoch":      epoch,
			"train_loss": trainLoss,
			"val_loss":   valLoss,
			"lr":         t.currentLR(),
		}
		t.fireEvent(callbacks.EventEpochEnd, epochMeta)

		fmt.Printf("[Epoch %3d/%d] train_loss=%.4f val_loss=%.4f lr=%.2e  (%s)\n",
			epoch+1, t.cfg.MaxEpochs, trainLoss, valLoss,
			t.currentLR(), time.Since(startTime).Round(time.Second))

		if t.cfg.MaxSteps > 0 && t.state.GlobalStep >= t.cfg.MaxSteps {
			break
		}
	}

	t.state.ElapsedTime = time.Since(startTime)
	t.fireEvent(callbacks.EventTrainEnd, nil)
	return nil
}

// ─── trainEpoch ──────────────────────────────────────────────────────────────

func (t *Trainer) trainEpoch(loader data.DataLoader) (float64, error) {
	t.model.Train(true)
	var totalLoss float64
	steps := 0

	t.optimizer.ZeroGrad()

	for batch := range loader.Iter() {
		select {
		case <-t.ctx.Done():
			return 0, context.Canceled
		default:
		}

		inputs, targets := batch.Inputs, batch.Targets
		// Move to device
		for i, inp := range inputs {
			inputs[i] = inp.To(t.device)
		}
		for i, tgt := range targets {
			targets[i] = tgt.To(t.device)
		}

		// Forward
		output := t.model.Forward(inputs...)

		// Loss (normalised by accumulation steps)
		loss := t.lossFn(output, targets[0])
		if t.cfg.GradAccumSteps > 1 {
			loss = loss.MulScalar(1.0 / float64(t.cfg.GradAccumSteps))
		}

		// NaN detection
		if t.cfg.DetectAnomalies {
			if math.IsNaN(loss.Item()) || math.IsInf(loss.Item(), 0) {
				return 0, fmt.Errorf("step %d: loss is %v", t.state.GlobalStep, loss.Item())
			}
		}

		// Backward
		loss.Backward(nil, false)

		// Gradient accumulation
		if (steps+1)%t.cfg.GradAccumSteps == 0 {
			// Grad clip
			if t.cfg.GradClipNorm > 0 {
				t.clipGradNorm(t.cfg.GradClipNorm)
			}
			t.optimizer.Step()
			t.optimizer.ZeroGrad()

			if t.scheduler != nil {
				t.scheduler.Step()
			}
			t.state.Update(loss.Item(), t.currentLR())
		}

		totalLoss += loss.Item()
		steps++

		if t.cfg.LogEvery > 0 && t.state.GlobalStep%t.cfg.LogEvery == 0 {
			smoothed := t.state.LastNLossMean(t.cfg.LogEvery)
			fmt.Printf("  step=%d loss=%.4f smooth=%.4f lr=%.2e\n",
				t.state.GlobalStep, loss.Item(), smoothed, t.currentLR())
		}

		if t.cfg.SaveEvery > 0 && t.state.GlobalStep%t.cfg.SaveEvery == 0 {
			path := fmt.Sprintf("%s/step_%d.pt", t.cfg.CheckpointDir, t.state.GlobalStep)
			_ = t.saveCheckpoint(path)
		}

		if t.cfg.MaxSteps > 0 && t.state.GlobalStep >= t.cfg.MaxSteps {
			break
		}
	}

	if steps == 0 {
		return 0, nil
	}
	return totalLoss / float64(steps), nil
}

// ─── evalEpoch ───────────────────────────────────────────────────────────────

func (t *Trainer) evalEpoch(loader data.DataLoader) (float64, error) {
	t.model.Train(false)
	var totalLoss float64
	steps := 0

	for batch := range loader.Iter() {
		inputs, targets := batch.Inputs, batch.Targets
		for i, inp := range inputs {
			inputs[i] = inp.To(t.device)
		}
		for i, tgt := range targets {
			targets[i] = tgt.To(t.device)
		}

		// No gradient during eval
		output := noGradForward(t.model, inputs...)
		loss := t.lossFn(output, targets[0])
		totalLoss += loss.Item()
		steps++
	}

	if steps == 0 {
		return 0, nil
	}
	return totalLoss / float64(steps), nil
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

func (t *Trainer) clipGradNorm(maxNorm float64) {
	var totalNorm float64
	for _, p := range t.model.Parameters() {
		g := p.Grad()
		if g == nil {
			continue
		}
		norm := g.Pow(2).Sum(-1, false).Sqrt().Item()
		totalNorm += norm * norm
	}
	totalNorm = math.Sqrt(totalNorm)
	clipCoef := maxNorm / (totalNorm + 1e-6)
	if clipCoef < 1.0 {
		for _, p := range t.model.Parameters() {
			g := p.Grad()
			if g == nil {
				continue
			}
			_ = g.MulScalar(clipCoef)
		}
	}
}

func (t *Trainer) currentLR() float64 {
	groups := t.optimizer.ParamGroups()
	if len(groups) == 0 {
		return 0
	}
	return groups[0].LR
}

func (t *Trainer) fireEvent(ev callbacks.Event, meta map[string]interface{}) {
	for _, cb := range t.callbacks {
		cb.OnEvent(ev, meta)
	}
}

func (t *Trainer) saveCheckpoint(path string) error {
	if t.model == nil {
		return fmt.Errorf("cannot checkpoint a nil model")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	modelPath := path
	if filepath.Ext(modelPath) != ".safetensors" {
		modelPath = strings.TrimSuffix(modelPath, filepath.Ext(modelPath)) + ".safetensors"
	}
	w := serialization.NewSTWriter()
	for name, param := range t.model.StateDict() {
		w.Add(name, param)
	}
	if err := w.Save(modelPath); err != nil {
		return fmt.Errorf("save model: %w", err)
	}
	meta := struct {
		Epoch      int     `json:"epoch"`
		GlobalStep int     `json:"global_step"`
		BestMetric float64 `json:"best_metric"`
		BestStep   int     `json:"best_step"`
		LR         float64 `json:"learning_rate"`
	}{t.state.Epoch, t.state.GlobalStep, t.state.BestMetric, t.state.BestStep, t.currentLR()}
	b, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(strings.TrimSuffix(modelPath, filepath.Ext(modelPath))+".json", b, 0644); err != nil {
		return err
	}
	return nil
}

func (t *Trainer) loadCheckpoint(path string) error {
	if t.model == nil {
		return fmt.Errorf("cannot restore into a nil model")
	}
	r, err := serialization.LoadST(path)
	if err != nil {
		return fmt.Errorf("load model: %w", err)
	}
	sd := make(map[string]*tensor.Tensor)
	for _, name := range r.Keys() {
		v, err := r.Get(name)
		if err != nil {
			return err
		}
		sd[name] = v
	}
	if err := t.model.LoadStateDict(sd); err != nil {
		return fmt.Errorf("load state dict: %w", err)
	}
	metaPath := strings.TrimSuffix(path, filepath.Ext(path)) + ".json"
	if b, err := os.ReadFile(metaPath); err == nil {
		var meta struct {
			Epoch      int     `json:"epoch"`
			GlobalStep int     `json:"global_step"`
			BestMetric float64 `json:"best_metric"`
			BestStep   int     `json:"best_step"`
		}
		if err := json.Unmarshal(b, &meta); err != nil {
			return err
		}
		t.state.Epoch, t.state.GlobalStep, t.state.BestMetric, t.state.BestStep = meta.Epoch, meta.GlobalStep, meta.BestMetric, meta.BestStep
	}
	return nil
}

func noGradForward(m nn.Module, inputs ...*tensor.Tensor) *tensor.Tensor {
	// In real impl: wraps with C.PyGoNoGrad guard
	return m.Forward(inputs...)
}

func (t *Trainer) Stop() { t.cancelFn() }

func (t *Trainer) State() TrainingState {
	t.state.mu.RLock()
	defer t.state.mu.RUnlock()
	return t.state
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
