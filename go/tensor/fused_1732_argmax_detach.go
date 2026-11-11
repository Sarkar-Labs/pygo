package tensor

import "fmt"

func Fused1732ArgmaxDetach(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("argmax_detach: nil operand")
	}
	return x.Argmax(0), nil
}
func Fused1732ArgmaxDetachName() string { return "argmax_detach" }
