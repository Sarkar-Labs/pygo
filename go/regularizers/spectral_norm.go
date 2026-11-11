package regularizers

import "time"

// GeneratedSpectralNorm records execution state for the spectral norm component.
type GeneratedSpectralNorm struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSpectralNorm() *GeneratedSpectralNorm {
	return &GeneratedSpectralNorm{Name: "spectral_norm", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSpectralNorm) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSpectralNorm) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
