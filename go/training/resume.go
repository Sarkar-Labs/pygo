package training

import "time"

// GeneratedResume records execution state for the resume component.
type GeneratedResume struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedResume() *GeneratedResume {
	return &GeneratedResume{Name: "resume", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedResume) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedResume) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
