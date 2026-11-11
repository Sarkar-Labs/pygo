package tensor

import "fmt"

func Fused1737MinClone(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("min_clone: nil operand")
	}
	return x, nil
}
func Fused1737MinCloneName() string { return "min_clone" }
