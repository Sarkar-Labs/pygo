package inference

import "time"

// GeneratedBadWords records execution state for the bad words component.
type GeneratedBadWords struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedBadWords() *GeneratedBadWords {
	return &GeneratedBadWords{Name: "bad_words", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedBadWords) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedBadWords) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
