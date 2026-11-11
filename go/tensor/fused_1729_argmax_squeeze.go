package tensor

import "fmt"

func Fused1729ArgmaxSqueeze(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("argmax_squeeze: nil operand")
	}
	return x.Argmax(0), nil
}
func Fused1729ArgmaxSqueezeName() string { return "argmax_squeeze" }
