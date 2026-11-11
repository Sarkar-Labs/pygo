package training

import "time"

// GeneratedValidationLoop records execution state for the validation loop component.
type GeneratedValidationLoop struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedValidationLoop() *GeneratedValidationLoop {
	return &GeneratedValidationLoop{Name: "validation_loop", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedValidationLoop) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedValidationLoop) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
