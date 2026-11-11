package tensor

import "fmt"

func Fused0141ReluSubCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_sub_ceil: nil operand")
	}
	r := x.Sub(y)
	r = r.ReLU()
	r = r.Ceil()
	return r, nil
}
func Fused0141ReluSubCeilName() string { return "relu_sub_ceil" }
