package tensor

import "fmt"

func Fused0003ReluDiv(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("relu_div: nil operand")
	}
	r := x.Div(y)
	r = r.ReLU()
	return r, nil
}
func Fused0003ReluDivName() string { return "relu_div" }
