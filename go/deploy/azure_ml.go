package deploy

import "time"

// GeneratedAzureMl records execution state for the azure ml component.
type GeneratedAzureMl struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedAzureMl() *GeneratedAzureMl {
	return &GeneratedAzureMl{Name: "azure_ml", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedAzureMl) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedAzureMl) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
