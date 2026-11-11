package tensor

import "fmt"

func Fused1731ArgmaxClone(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("argmax_clone: nil operand")
	}
	return x.Argmax(0), nil
}
func Fused1731ArgmaxCloneName() string { return "argmax_clone" }
