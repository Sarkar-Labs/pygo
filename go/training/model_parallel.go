package training

import "time"

// GeneratedModelParallel records execution state for the model parallel component.
type GeneratedModelParallel struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedModelParallel() *GeneratedModelParallel {
	return &GeneratedModelParallel{Name: "model_parallel", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedModelParallel) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedModelParallel) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
