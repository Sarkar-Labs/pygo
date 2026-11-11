package transforms

import "time"

// GeneratedTokenize records execution state for the tokenize component.
type GeneratedTokenize struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedTokenize() *GeneratedTokenize {
	return &GeneratedTokenize{Name: "tokenize", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedTokenize) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedTokenize) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
