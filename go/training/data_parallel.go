package training

import "time"

// GeneratedDataParallel records execution state for the data parallel component.
type GeneratedDataParallel struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDataParallel() *GeneratedDataParallel {
	return &GeneratedDataParallel{Name: "data_parallel", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDataParallel) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDataParallel) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
