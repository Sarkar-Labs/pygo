package models

import "time"

// GeneratedDinoCoreml records execution state for the dino coreml component.
type GeneratedDinoCoreml struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDinoCoreml() *GeneratedDinoCoreml {
	return &GeneratedDinoCoreml{Name: "dino_coreml", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDinoCoreml) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDinoCoreml) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
