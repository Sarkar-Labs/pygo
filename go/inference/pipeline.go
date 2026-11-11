package inference

import "time"

// GeneratedPipeline records execution state for the pipeline component.
type GeneratedPipeline struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedPipeline() *GeneratedPipeline {
	return &GeneratedPipeline{Name: "pipeline", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedPipeline) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedPipeline) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
