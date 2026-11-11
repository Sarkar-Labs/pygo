package inference

import "time"

// GeneratedForceWords records execution state for the force words component.
type GeneratedForceWords struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedForceWords() *GeneratedForceWords {
	return &GeneratedForceWords{Name: "force_words", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedForceWords) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedForceWords) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
