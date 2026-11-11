package inference

import "time"

// GeneratedNoRepeatNgram records execution state for the no repeat ngram component.
type GeneratedNoRepeatNgram struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedNoRepeatNgram() *GeneratedNoRepeatNgram {
	return &GeneratedNoRepeatNgram{Name: "no_repeat_ngram", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedNoRepeatNgram) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedNoRepeatNgram) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
