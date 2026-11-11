package tensor

import "fmt"

func Fused1538Log2DivSigmoid(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_div_sigmoid: nil operand")
	}
	r := x.Div(y)
	r = r.Log2()
	r = r.Sigmoid()
	return r, nil
}
func Fused1538Log2DivSigmoidName() string { return "log2_div_sigmoid" }
