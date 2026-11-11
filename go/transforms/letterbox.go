package transforms

import "time"

// GeneratedLetterbox records execution state for the letterbox component.
type GeneratedLetterbox struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedLetterbox() *GeneratedLetterbox {
	return &GeneratedLetterbox{Name: "letterbox", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedLetterbox) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedLetterbox) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
