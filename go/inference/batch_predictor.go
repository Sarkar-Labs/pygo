package inference

import "time"

// GeneratedBatchPredictor records execution state for the batch predictor component.
type GeneratedBatchPredictor struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedBatchPredictor() *GeneratedBatchPredictor {
	return &GeneratedBatchPredictor{Name: "batch_predictor", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedBatchPredictor) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedBatchPredictor) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
