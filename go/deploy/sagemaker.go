package deploy

import "time"

// GeneratedSagemaker records execution state for the sagemaker component.
type GeneratedSagemaker struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSagemaker() *GeneratedSagemaker {
	return &GeneratedSagemaker{Name: "sagemaker", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSagemaker) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSagemaker) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
