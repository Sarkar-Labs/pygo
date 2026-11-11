package inference

import "time"

// GeneratedStreamPredictor records execution state for the stream predictor component.
type GeneratedStreamPredictor struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedStreamPredictor() *GeneratedStreamPredictor {
	return &GeneratedStreamPredictor{Name: "stream_predictor", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedStreamPredictor) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedStreamPredictor) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
