package tensor

import "fmt"

func Fused0108ReluDivAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_div_abs: nil operand")
	}
	r := x.Div(y)
	r = r.ReLU()
	r = r.Abs()
	return r, nil
}
func Fused0108ReluDivAbsName() string { return "relu_div_abs" }
