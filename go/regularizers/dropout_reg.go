package regularizers

import "time"

// GeneratedDropoutReg records execution state for the dropout reg component.
type GeneratedDropoutReg struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDropoutReg() *GeneratedDropoutReg {
	return &GeneratedDropoutReg{Name: "dropout_reg", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDropoutReg) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDropoutReg) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
