package callbacks

import "time"

// GeneratedOnBatchBegin records execution state for the on batch begin component.
type GeneratedOnBatchBegin struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedOnBatchBegin() *GeneratedOnBatchBegin {
	return &GeneratedOnBatchBegin{Name: "on_batch_begin", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedOnBatchBegin) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedOnBatchBegin) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
