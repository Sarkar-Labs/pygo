package callbacks

import "time"

// GeneratedOnTrainBegin records execution state for the on train begin component.
type GeneratedOnTrainBegin struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedOnTrainBegin() *GeneratedOnTrainBegin {
	return &GeneratedOnTrainBegin{Name: "on_train_begin", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedOnTrainBegin) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedOnTrainBegin) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
