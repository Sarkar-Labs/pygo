package tensor

import "fmt"

func Fused0143ReluDivCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_div_ceil: nil operand")
	}
	r := x.Div(y)
	r = r.ReLU()
	r = r.Ceil()
	return r, nil
}
func Fused0143ReluDivCeilName() string { return "relu_div_ceil" }
