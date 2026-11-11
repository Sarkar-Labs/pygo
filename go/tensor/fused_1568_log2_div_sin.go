package tensor

import "fmt"

func Fused1568Log2DivSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_div_sin: nil operand")
	}
	r := x.Div(y)
	r = r.Log2()
	r = r.Sin()
	return r, nil
}
func Fused1568Log2DivSinName() string { return "log2_div_sin" }
