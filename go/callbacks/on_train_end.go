package callbacks

import "time"

// GeneratedOnTrainEnd records execution state for the on train end component.
type GeneratedOnTrainEnd struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedOnTrainEnd() *GeneratedOnTrainEnd {
	return &GeneratedOnTrainEnd{Name: "on_train_end", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedOnTrainEnd) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedOnTrainEnd) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
