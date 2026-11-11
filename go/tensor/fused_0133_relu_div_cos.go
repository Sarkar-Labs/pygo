package tensor

import "fmt"

func Fused0133ReluDivCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_div_cos: nil operand")
	}
	r := x.Div(y)
	r = r.ReLU()
	r = r.Cos()
	return r, nil
}
func Fused0133ReluDivCosName() string { return "relu_div_cos" }
