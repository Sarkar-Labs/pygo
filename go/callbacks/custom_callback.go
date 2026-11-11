package callbacks

import "time"

// GeneratedCustomCallback records execution state for the custom callback component.
type GeneratedCustomCallback struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedCustomCallback() *GeneratedCustomCallback {
	return &GeneratedCustomCallback{Name: "custom_callback", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedCustomCallback) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedCustomCallback) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
