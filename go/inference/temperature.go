package inference

import "time"

// GeneratedTemperature records execution state for the temperature component.
type GeneratedTemperature struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedTemperature() *GeneratedTemperature {
	return &GeneratedTemperature{Name: "temperature", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedTemperature) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedTemperature) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
