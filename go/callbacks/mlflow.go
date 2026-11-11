package callbacks

import "time"

// GeneratedMlflow records execution state for the mlflow component.
type GeneratedMlflow struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMlflow() *GeneratedMlflow {
	return &GeneratedMlflow{Name: "mlflow", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMlflow) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMlflow) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
