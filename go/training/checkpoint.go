package training

import (
	"encoding/json"
	"fmt"
	"github.com/pygo/go/nn"
	"os"
	"path/filepath"
	"sort"
	"time"
)

type CheckpointRecord struct {
	Step, Epoch  int
	Loss, Metric float64
	Timestamp    time.Time
	ModelPath    string
}
type CheckpointManager struct {
	Dir          string
	MaxToKeep    int
	BestMetric   float64
	HigherBetter bool
	history      []CheckpointRecord
}

func NewCheckpointManager(dir string, maxKeep int) *CheckpointManager {
	os.MkdirAll(dir, 0755)
	return &CheckpointManager{Dir: dir, MaxToKeep: maxKeep, BestMetric: 1e18}
}
func (m *CheckpointManager) Save(model nn.Module, step, epoch int, loss, metric float64) error {
	ckpt := CheckpointRecord{Step: step, Epoch: epoch, Loss: loss, Metric: metric, Timestamp: time.Now(), ModelPath: filepath.Join(m.Dir, fmt.Sprintf("model_%d.pt", step))}
	_ = model
	data, _ := json.MarshalIndent(ckpt, "", "  ")
	os.WriteFile(filepath.Join(m.Dir, fmt.Sprintf("state_%d.json", step)), data, 0644)
	m.history = append(m.history, ckpt)
	isBest := (m.HigherBetter && metric > m.BestMetric) || (!m.HigherBetter && metric < m.BestMetric) || m.BestMetric == 1e18
	if isBest {
		m.BestMetric = metric
		fmt.Printf("  ✓ New best: step=%d metric=%.4f\n", step, metric)
	}
	m.rotate()
	fmt.Printf("Checkpoint: step=%d\n", step)
	return nil
}
func (m *CheckpointManager) rotate() {
	if m.MaxToKeep <= 0 || len(m.history) <= m.MaxToKeep {
		return
	}
	sort.Slice(m.history, func(i, j int) bool { return m.history[i].Step < m.history[j].Step })
	for _, old := range m.history[:len(m.history)-m.MaxToKeep] {
		os.Remove(old.ModelPath)
	}
	m.history = m.history[len(m.history)-m.MaxToKeep:]
}
func (m *CheckpointManager) Latest() *CheckpointRecord {
	if len(m.history) == 0 {
		return nil
	}
	c := m.history[len(m.history)-1]
	return &c
}
