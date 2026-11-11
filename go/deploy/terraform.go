package deploy

import "time"

// GeneratedTerraform records execution state for the terraform component.
type GeneratedTerraform struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedTerraform() *GeneratedTerraform {
	return &GeneratedTerraform{Name: "terraform", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedTerraform) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedTerraform) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
