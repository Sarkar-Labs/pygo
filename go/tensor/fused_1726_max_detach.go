package tensor

import "fmt"

func Fused1726MaxDetach(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("max_detach: nil operand")
	}
	return x.Max(0, false), nil
}
func Fused1726MaxDetachName() string { return "max_detach" }
