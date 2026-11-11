package tensor

import "fmt"

func Fused1533Log2DivRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_div_relu: nil operand")
	}
	r := x.Div(y)
	r = r.Log2()
	r = r.ReLU()
	return r, nil
}
func Fused1533Log2DivReluName() string { return "log2_div_relu" }
