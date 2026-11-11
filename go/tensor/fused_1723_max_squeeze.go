package tensor

import "fmt"

func Fused1723MaxSqueeze(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("max_squeeze: nil operand")
	}
	return x.Max(0, false), nil
}
func Fused1723MaxSqueezeName() string { return "max_squeeze" }
