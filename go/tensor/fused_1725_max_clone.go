package tensor

import "fmt"

func Fused1725MaxClone(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("max_clone: nil operand")
	}
	return x.Max(0, false), nil
}
func Fused1725MaxCloneName() string { return "max_clone" }
