package callbacks

import "time"

// GeneratedEarlyStopping records execution state for the early stopping component.
type GeneratedEarlyStopping struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedEarlyStopping() *GeneratedEarlyStopping {
	return &GeneratedEarlyStopping{Name: "early_stopping", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedEarlyStopping) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedEarlyStopping) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
