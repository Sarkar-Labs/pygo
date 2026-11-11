package transforms

import "time"

// GeneratedGaussianBlur records execution state for the gaussian blur component.
type GeneratedGaussianBlur struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGaussianBlur() *GeneratedGaussianBlur {
	return &GeneratedGaussianBlur{Name: "gaussian_blur", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGaussianBlur) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGaussianBlur) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
