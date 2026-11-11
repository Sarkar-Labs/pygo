package tensor

import "fmt"

func Fused1730ArgmaxUnsqueeze(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("argmax_unsqueeze: nil operand")
	}
	return x.Argmax(0), nil
}
func Fused1730ArgmaxUnsqueezeName() string { return "argmax_unsqueeze" }
