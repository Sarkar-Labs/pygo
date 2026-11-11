package callbacks

import "time"

// GeneratedOnBatchEnd records execution state for the on batch end component.
type GeneratedOnBatchEnd struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedOnBatchEnd() *GeneratedOnBatchEnd {
	return &GeneratedOnBatchEnd{Name: "on_batch_end", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedOnBatchEnd) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedOnBatchEnd) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
