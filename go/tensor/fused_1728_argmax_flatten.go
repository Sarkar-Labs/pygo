package tensor

import "fmt"

func Fused1728ArgmaxFlatten(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("argmax_flatten: nil operand")
	}
	return x.Argmax(0), nil
}
func Fused1728ArgmaxFlattenName() string { return "argmax_flatten" }
